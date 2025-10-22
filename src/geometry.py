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


def sdf_to_tsdf(sdf, thresh):
    """
    Convert a signed distance function to a truncated signed distance function.
    This is a general purpose function for SDFs in any dimension

    Parameters
    ----------
    sdf: ndarray (...)
        A signed distance function
    thresh: float
        Distance threshold beyond which to exclude data
    
    Returns
    -------
    tsdf: ndarray (...)
        The truncated signed distance function.  NaNs indicate data outside
        of the scanned range
    """
    tsdf = np.nan*np.ones_like(sdf)
    sel = np.abs(sdf) < thresh
    tsdf[sel] = sdf[sel]
    return tsdf


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

def get_sdf_2d(X, N, G):
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

def get_curv_vectors(X, MaxOrder, sigma, loop = False, m = 'nearest'):
    """
    Get smoothed curvature vectors up to a particular order
    
    Parameters
    ----------
    X: ndarray(N, d)
        An N x d matrix of points in R^d
    MaxOrder: int
        The maximum order of torsion to compute (e.g. 3 for position, velocity, and curvature, and torsion)
    sigma: float
        The smoothing amount
    loop: boolean
        Whether to treat this trajectory as a topological loop (i.e. add an edge between first and last point)
    
    Returns
    -------
    Curvs: A list of (N, 3) arrays, starting with the smoothed curve, then followed
           by the smoothed velocity, curvature, torsion, etc. up to the MaxOrder
    """
    from scipy.ndimage.filters import gaussian_filter1d as gf1d
    if loop:
        m = 'wrap'
    XSmooth = gf1d(X, sigma, axis=0, order = 0, mode = m)
    Vel = gf1d(X, sigma, axis=0, order = 1, mode = m)
    VelNorm = np.sqrt(np.sum(Vel**2, 1))
    VelNorm[VelNorm == 0] = 1
    Curvs = [XSmooth, Vel]
    for order in range(2, MaxOrder+1):
        Tors = gf1d(X, sigma, axis=0, order=order, mode = m)
        for j in range(1, order):
            #Project away other components
            NormsDenom = np.sum(Curvs[j]**2, 1)
            NormsDenom[NormsDenom == 0] = 1
            Norms = np.sum(Tors*Curvs[j], 1)/NormsDenom
            Tors = Tors - Curvs[j]*Norms[:, None]
        Tors = Tors/(VelNorm[:, None]**order)
        Curvs.append(Tors)
    return Curvs


def reconstruct_signed_voronoi_2d(X, c):
    """
    Perform a signed Voronoi reconstruction of a 2D curve

    Parameters
    ----------
    X: ndarray(N, 3)
        Location of Voronoi sites
    c: ndarray(N)
        0/1 inside/outside designation of each site
    
    Returns
    -------
    centers: ndarray(n_centers, 3)
        Locations of Voronoi vertices (i.e. Delaunay circumcenters)
    edges: ndarray(n_edges, 2, dtype=int)
        Edges between centers
    """
    from scipy.spatial import Delaunay
    tri = Delaunay(X)
    simplices = tri.simplices

    ## Step 1: Figure out geometry
    centers = get_circumcenters(X, simplices)

    ## Step 2: Figure out topology
    # Need to be able to quickly look up two involved triangles for an edge
    edge2tris = {}
    for i, simplex in enumerate(simplices):
        for j in range(len(simplex)):
            edge = [simplex[j], simplex[(j+1)%len(simplex)]]
            edge = tuple(sorted(edge))
            if not edge in edge2tris:
                edge2tris[edge] = []
            edge2tris[edge].append(i)
    edges = []
    for (e1, e2), tris in edge2tris.items():
        # Only include edges between inside/outside, as well as non-boundary edges
        if c[e1] != c[e2] and len(tris) == 2:
            tri1, tri2 = tris
            edges.append([tri1, tri2])
    edges = np.array(edges, dtype=int)
    
    return centers, edges



######################################################
#              3D IMPLICIT UTILITIES
######################################################

def mask3d_to_gridpos(mask, xs, ys, zs):
    """
    Parameters
    ----------
    mask: ndarray(K, dtype=int)
        Raveled indices into the sparse M1 x M2 x M3 grid implied 
        by xs, ys, and zs at which to compute the sdf before truncating 
        to the tsdf
    xs: ndarray(M1)
        x locations on a grid on which to sample the implicit function
    ys: ndarray(M2)
        y locations on a grid on which to sample the implicit function
    zs: ndarray(M3)
        z locations on a grid on which to sample the implicit function
    
    Returns
    -------
    ndarray(K, 3)
        XYZ locations of grid points
    """
    xyz_idx = np.unravel_index(mask, (xs.size, ys.size, zs.size))
    xyz_idx = np.array(xyz_idx).T
    xyz = np.zeros(xyz_idx.shape)
    for k, w in enumerate([xs, ys, zs]):
        xyz[:, k] = w[xyz_idx[:, k]]
    return xyz

def get_tsdf_3d_masked(X, N, xs, ys, zs, thresh, mask, shrink_mask=True):
    """
    Compute a discretized truncated signed distance function (TSDF)
    restricted to a mask for a point cloud in 3D

    Parameters
    ----------
    X: ndarray(N, 3)
        A point cloud in R^2
    N: ndarray(N, 3)
        The corresponding normals for X
    xs: ndarray(M1)
        x locations on a grid on which to sample the implicit function
    ys: ndarray(M2)
        y locations on a grid on which to sample the implicit function
    zs: ndarray(M3)
        z locations on a grid on which to sample the implicit function
    thresh: float
        Threshold distance at which to cutoff the tsdf
    mask: ndarray(K, dtype=int)
        Raveled indices into the sparse M1 x M2 x M3 grid implied 
        by xs, ys, and zs at which to compute the sdf before truncating 
        to the tsdf
    shrink_mask: boolean
        If true, shrink the mask to only include voxels that have a
        distance that's within thresh.  If false, compute the sdf
        on the entire mask
    
    Returns
    -------
    xyz: ndarray(K, 3, dtype=float)
        Coordinates of the tsdf
    tsdf: ndarray(K, dtype=float)
        The signed implicit function for the surface
    """
    from sklearn.neighbors import NearestNeighbors
    ## Step 1: Convert mask into xyz grid locations based on xs, ys, zs
    xyz = mask3d_to_gridpos(mask, xs, ys, zs)
    ## Step 2: Use a tree-based data structure to quickly query
    ## nearest neighbors in the scan to all points on the grid
    neighb = NearestNeighbors(n_neighbors=1)
    neighb.fit(X)
    dists_i, neighbs_idx = neighb.kneighbors(xyz, 1, return_distance=True)
    neighbs_idx = neighbs_idx.flatten()
    dists_i = dists_i.flatten()
    ## Step 3: Compute the sdf at the grid points that are within
    ## a (possibly shrunken) mask
    if shrink_mask:
        neighbs_idx = neighbs_idx[dists_i < thresh]
        xyz = xyz[dists_i < thresh]
        dists_i = dists_i[dists_i < thresh]
    grid_nn = X[neighbs_idx, :]
    grid_normals = N[neighbs_idx, :]
    dX = xyz - grid_nn
    tsdf = np.sign(np.sum(dX*grid_normals, axis=1))*dists_i
    return xyz, tsdf

def get_aabbox3d_mask(xs, ys, zs, mins, maxes):
    """
    Compute a mask based on an axis-aligned bounding box
    TODO: Add capability for non axis aligned bounding boxes

    Parameters
    ----------
    xs: ndarray(M1)
        x locations on a grid on which to sample the implicit function
    ys: ndarray(M2)
        y locations on a grid on which to sample the implicit function
    zs: ndarray(M3)
        z locations on a grid on which to sample the implicit function
    mins: [float, float float]
        Bottom left of AABB
    maxes: [float, float]
        Upper right of AABB

    Returns
    -------
    mask: ndarray(K, dtype=int)
        The raveled indices into the sparse M1 x M2 x M3 grid at which
        the tsdf values are specified.
    """
    # Figure out indices of this aabb into full grid
    idxs = []
    for k, w in enumerate([xs, ys, zs]):
        min, max = mins[k], maxes[k]
        idx = np.arange(w.size)[(w >= min)*(w <= max)]
        idxs.append(idx)
    x, y, z = [w.flatten() for w in np.meshgrid(*idxs, indexing='ij')]
    return np.ravel_multi_index((x, y, z), (xs.size, ys.size, zs.size))


def get_tsdf_3d(X, N, xs, ys, zs, thresh):
    """
    Compute a discretized truncated signed distance function (TSDF) for a point 
    cloud in 3D.  For speed, narrow down to the voxels in the point cloud's
    axis-aligned bounding box expanded by "thresh"

    Parameters
    ----------
    X: ndarray(N, 3)
        A point cloud in R^2
    N: ndarray(N, 3)
        The corresponding normals for X
    xs: ndarray(M1)
        x locations on a grid on which to sample the implicit function
    ys: ndarray(M2)
        y locations on a grid on which to sample the implicit function
    zs: ndarray(M3)
        z locations on a grid on which to sample the implicit function
    thresh: float
        Threshold distance at which to cutoff the tsdf
    
    Returns
    -------
    xyz: ndarray(K, 3, dtype=float)
        Coordinates of the tsdf
    tsdf: ndarray(K, dtype=float)
        The signed implicit function for the surface
    """
    # Narrow down mask to an axis-aligned bounding box expanded by thresh
    mins = np.min(X, axis=0) - thresh
    maxes = np.max(X, axis=0) + thresh
    mask = get_aabbox3d_mask(xs, ys, zs, mins, maxes)
    return get_tsdf_3d_masked(X, N, xs, ys, zs, thresh, mask)



######################################################
#                  3D I/O UTILITIES
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