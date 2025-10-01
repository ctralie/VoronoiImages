import numpy as np

def get_circumcenters(X, simplices):
    """
    Compute the circumcenters of a bunch of simplices on a Euclidean point set

    Parameters
    ----------
    X: ndarray(N, d)
        N points in d dimensions
    simplices: ndarray(n_simplices, d+1)
        Indices of simplices
    
    Returns
    -------
    ndarray(n_simplices, d)
        Center locations
    """
    XSimplices = X[simplices, :]
    mu = np.mean(XSimplices, axis=1, keepdims=True) # Center for numerical precision
    XSimplices -= mu
    d = X.shape[1]
    
    denom = np.ones((simplices.shape[0], d+1, d+1))
    denom[:, :, 0:d] = XSimplices
    denom = 2*np.linalg.det(denom)
    
    centers = np.zeros((simplices.shape[0], d))
    for k in range(d):
        num = np.ones((simplices.shape[0], d+1, d+1))
        num[:, :, 0:d] = XSimplices
        num[:, :, k] = np.sum(num[:, :, 0:d]**2, axis=-1)
        centers[:, k] = np.linalg.det(num)/denom
    centers += mu[:, 0, :]

    return centers

def estimate_scan_normals(X, pos, k=10):
    """
    Compute the normals at every point of a scan, using the camera
    position and towards vector to orient them towards the outside
    of the scan

    Parameters
    ----------
    X: ndarray(n_points, d):
        Point clouds of the scan, in world coordinates
    pos: ndarray(d)
        The position of the camera that measured this scan, in 
        world coordinates
    k: int
        The number of nearest neighbors to use to fit a plane
    
    Returns
    -------
    N: ndarray(n_points, d)
        The normals, oriented towards the camera
    """
    from scipy import spatial
    from sklearn.decomposition import PCA
    pca = PCA(X.shape[1])
    k = max(k, X.shape[1])
    tree = spatial.KDTree(X)
    N = np.zeros_like(X)
    for i in range(X.shape[0]):
        _, idxs = tree.query(X[i, :], k+1)
        Xi = X[idxs, :]
        pca.fit(Xi)
        Ni = pca.components_[-1, :]
        if np.sum(Ni*(pos-Xi)) < 0:
            # Want normals facing towards the camera
            Ni *= -1
        N[i, :] = Ni
    return N


######################################################
#                  2D UTILITIES
######################################################

def get_image_curve(filename, isolevel=0.8):
    """
    Load a curve in as the levelset of some grayscale image
    
    Parameters
    ----------
    filename: string
        Path to image file
    isolevel: float
        Levelset isolevel to use

    Returns
    -------
    ndarray(N, 2)
        Largest connected component at the isolevel
    """
    from skimage import io, measure
    I = io.imread(filename)
    # Convert to grayscale
    I = 0.3*I[:, :, 0] + 0.59*I[:, :, 1] + 0.11*I[:, :, 2]
    I = I/255
    contours = measure.find_contours(I, isolevel)
    largest = np.array([[]])
    if len(contours) > 0:
        sizes = np.array([c.shape[0] for c in contours])
        largest = contours[np.argmax(sizes)]
    return np.flipud(largest)


def estimate_normals_curve(X, sigma=2, loop=False):
    """
    Estimate the normals at every connected curve component

    Parameters
    ----------
    X: ndarray(N, 2)
        A connected set of points in R^2
    sigma: float
        How much to smooth the normal estimates
    loop: boolean
        Whether to treat this trajectory as a topological loop (i.e. add an edge between first and last point)
    
    Returns
    -------
    normals: ndarray(N, 2)
        Curve normals
    """
    from scipy.ndimage import convolve
    num_points = X.shape[0]
    t = np.arange(-3*sigma, 3*sigma+1)
    kernel = np.exp(-t**2/(2*sigma**2))
    kernel /= np.sum(kernel)
    if loop:
        X = np.concatenate((X, X), axis=0)
    N = X[1::, :] - X[0:-1, :]
    N[:, 0] = convolve(N[:, 0], kernel)
    N[:, 1] = convolve(N[:, 1], kernel)
    if loop:
        N = N[0:num_points, :]
    else:
        N = np.concatenate((N, np.reshape(N[-1, :], (1, N.shape[1]))))
    N = np.array([-N[:, 1], N[:, 0]]).T
    norms = np.sum(N**2, axis=1)
    norms[norms <= 0] = 0
    norms = np.sqrt(norms)
    N = -N/norms[:, None]
    return N

def clamp(arrs, boolexpr):
    """
    Clamp numpy arrays to a boolean expression
    Parameters
    ----------
    arrs: Length-k list of [ndarray(N)]
        List of arrays
    boolexpr: ndarray(N, dtype=np.boolean)
        Boolean expression
    Returns
    -------
    Length-k list of [ndarray(M)], where M is
    the number of true evaluations in boolexpr
    """
    return [arr[boolexpr] for arr in arrs]

def ray_intersect_curve(p0, v, X):
    """
    Parameters
    ----------
    p0: ndarray(2)
        Initial point on ray
    v: ndarray(2)
        Direction of ray
    X: ndarray(N, 2)
        Points on the curve
    
    Returns {'t': float,
                    The distance of the closest point of intersection, or 
                    np.inf if there was no intersection
            'p': ndarray(2)
                    Point of intersection
            }
    """
    # vec{a} + s*vec{u} refers to line segments on curve
    # vec{b} + t*vec{v} refers to ray
    u = X[1::, :] - X[0:-1, :]
    u = np.concatenate((u, [u[-1, :]]), axis=0) # Continue the last slope
    # Use Cramer's rule
    # uy*vx - ux*vy
    denom = u[:, 1]*v[0] - u[:, 0]*v[1] 
    # vx*(by - ay) - vy*(bx - ax)
    num_s = v[0]*(p0[1]-X[:, 1]) - v[1]*(p0[0]-X[:, 0])
    # ux*(by - ay) - uy*(bx - ax)
    num_t = u[:, 0]*(p0[1]-X[:, 1]) - u[:, 1]*(p0[0]-X[:, 0])
    
    idx = np.arange(X.shape[0])
    [num_s, num_t, denom, idx] = clamp([num_s, num_t, denom, idx], np.abs(denom) > 0)
    s = num_s/denom
    t = num_t/denom
    # Intersection is within the bounds of the segments
    [s, t, idx] = clamp([s, t, idx], (s >= 0)*(s <= 1))
    tmin = np.inf
    p = np.zeros(2)
    if s.size > 0:
        [s, t, idx] = clamp([s, t, idx], t >= 0)
        if t.size > 0:
            minidx = np.argmin(t)
            tmin = t[minidx]
            s = s[minidx]
            t = t[minidx]
            idx = idx[minidx]
            p = p0 + t*v
    return {'t':tmin, 'p':p}

def get_sdf(X, N, G):
    """
    Compute a discretized distance function of a set of connected
    components on a sampled curve

    Parameters
    ----------
    X: ndarray(N, 2)
        A point cloud in R^2
    N: ndarray(N, 2)
        The corresponding normals for X
    G: ndarray(M, 2)
        Grid locations at which to sample sdf
    Returns
    -------
    sdf: ndarray(M)
        The signed implicit function for the curve
    """
    from scipy import spatial
    # Construct a KD tree on the curve so we can find nearest
    # neighbors for every grid point
    M = G.shape[0]
    tree = spatial.KDTree(X)
    dists_i, neighbs_idx = tree.query(G, 1)
    neighbs_idx = neighbs_idx.flatten()
    grid_nn = X[neighbs_idx, :]
    grid_normals = N[neighbs_idx, :]
    dX = G - grid_nn
    return np.sign(np.sum(dX*grid_normals, axis=1))*dists_i

######################################################
#                  3D UTILITIES
######################################################

def load_off_file(filename):
    """
    Load in a 3D mesh in OFF format

    Parameters
    ---------
    filename: string
        Path to file to load
    
    Returns
    -------
    {
        'VPos': ndarray(N, 3)
            Positions of vertices
        'ITris': ndarray(T, 3)
            Triangles, indexed into VPos
        'VColors': ndarray(N, 3)
            Colors associated to each vertex
    }
    """
    fin = open(filename, 'r')
    nVertices = 0
    nFaces = 0
    lineCount = 0
    face = 0
    vertex = 0
    divideColor = False
    VPos = []
    VColors = []
    ITris = []
    for line in fin:
        lineCount = lineCount+1
        fields = line.split() #Splits whitespace by default
        if len(fields) == 0: #Blank line
            continue
        if fields[0][0] in ['#', '\0', ' '] or len(fields[0]) == 0:
            continue
        #Check section
        if nVertices == 0:
            if fields[0] == "OFF" or fields[0] == "COFF":
                if len(fields) > 2:
                    fields[1:4] = [int(field) for field in fields]
                    [nVertices, nFaces] = fields[1:3]  
                if fields[0] == "COFF":
                    divideColor = True            
            else:
                fields[0:3] = [int(field) for field in fields]
                [nVertices, nFaces] = fields[0:2]
            #Pre-allocate vertex arrays    
            VPos = np.zeros((nVertices, 3)) 
            VColors = np.zeros((nVertices, 3))
            ITris = np.zeros((nFaces, 3), dtype=int)
        elif vertex < nVertices:
            VPos[vertex, :] = [float(i) for i in fields[0:3]]
            color = np.array([0.5, 0.5, 0.5]) #Gray by default
            if len(fields) >= 6:
                #There is color information
                if divideColor:
                    color = [float(c)/255.0 for c in fields[3:6]]
                else:
                    color = [float(c) for c in fields[3:6]]
            VColors[vertex, :] = color
            vertex = vertex+1
        elif face < nFaces:
            #Assume the vertices are specified in CCW order
            ITris[face, :] = [int(i) for i in fields[1::]]
            face = face+1
    fin.close()
    if np.max(VColors) > 1:
        #Rescale colors
        VColors = VColors / 255.0
    return dict(VPos=VPos, ITris=ITris, VColors=VColors)

def save_off_file(filename, VPos, ITris):
    """
    Save OFF file given vertex and triangle buffers
    
    Parameters
    ----------
    filename: string
        Path to which to save file
    VPos: ndarray(M, 3)
        Coordinates of the M vertices
    ITris: ndarray(N, 3)
        Vertex indices of the N triangles
    """
    nV = VPos.shape[0]
    nF = ITris.shape[0]
    if nV == 0:
        print("ERROR: The mesh you're trying to save has zero vertices, so the volume is either all negative or all positive")
        return
    fout = open(filename, "w")
    fout.write("OFF\n%i %i %i\n"%(nV, nF, 0))
    for i in range(nV):
        fout.write("%g %g %g\n"%tuple(VPos[i, :]))
    for i in range(nF):
        fout.write("3 %i %i %i\n"%tuple(ITris[i, :]))
    fout.close()