import numpy as np
import matplotlib.pyplot as plt
import time

def get_greedy_perm(X, n_perm=None, dpoint2all=lambda u, i:np.sqrt(np.sum((u - u[i, :][None, :])**2, axis=1))):
    """
    Compute a furthest point sampling permutation of a set of points
    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of either data
    n_perm: int
        Number of points to take in the permutation
        feature array
    Returns
    -------
    idx_perm: ndarray(n_perm)
        Indices of points in the greedy permutation
    lambdas: ndarray(n_perm)
        Covering radii at different points
    """
    if not n_perm:
        n_perm = X.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    idx_perm = np.zeros(n_perm, dtype=np.int64)
    lambdas = np.zeros(n_perm)
    ds = dpoint2all(X, 0)
    dperm2all = [ds]
    for i in range(1, n_perm):
        idx = np.argmax(ds)
        idx_perm[i] = idx
        lambdas[i - 1] = ds[idx]
        dperm2all.append(dpoint2all(X, idx))
        ds = np.minimum(ds, dperm2all[-1])
    lambdas[-1] = np.max(ds)
    return idx_perm, lambdas

def sobel(IGray):
    """
    Do a multidimensional sobel filter on a voxel grid

    Parameters
    ----------
    IGray: ndarray(M, N, ...)
        Voxel grid of grayscale values
    
    Returns
    -------
    ndarray(M, N, ...)
        Sobel filter gradient magnitudes
    """
    from scipy.ndimage import convolve1d
    derivs = np.zeros(tuple(list(IGray.shape)+[len(IGray.shape)]))
    dims = len(IGray.shape)
    for i in range(dims):
        derivs[..., i] = IGray
        for k in range(dims):
            kernel = np.array([1, 2, 1], dtype=float)
            if i == k:
                kernel = np.array([1, 0, -1], dtype=float)
            derivs[..., i] = convolve1d(derivs[..., i], kernel, axis=k, mode='nearest')
    return np.sqrt(np.sum(derivs**2, axis=-1))

def splat_voronoi_image_1nn(Y, M, N):
    """
    Splat a voronoi image on a grid of an arbitrary size

    Parameters
    ----------
    Y: ndarray(N, 5)
        xyrgb coordinates
    M: int
        Number of rows
    N: int
        Number of columns
    
    Returns
    -------
    ndarray(M, N, 3)
        Voronoi image
    """
    from scipy.spatial import KDTree
    xpix, ypix = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, M), indexing='xy')
    xpix = xpix.flatten()
    ypix = ypix.flatten()
    X = np.array([xpix, ypix]).T

    tree = KDTree(Y[:, 0:2])
    _, idx = tree.query(X, k=1, workers=-1) # Nearest landmark indices to pixels
    idx = idx.flatten()
    J = Y[idx, 2::]
    J = np.reshape(J, (M, N, 3))
    return J

def stochastic_universal_sample(weights, target_points, jitter=0):
    """
    Sample pixels according to a particular density using 
    stochastic universal sampling

    Parameters
    ----------
    weights: ndarray(M, N, ...)
        The weights of each voxel, in the range [0, 1]
    target_points: int
        The number of desired samples
    jitter: float
        Perform a jitter with this standard deviation of a voxel
    
    Returns
    -------
    uv: ndarray(N, dim)
        Location of point samples, in units of pixels
    """
    choices = np.zeros(target_points, dtype=int)
    w = np.zeros(weights.size+1)
    order = np.random.permutation(weights.size)
    w[1::] = weights.flatten()[order]
    w = w/np.sum(w)
    w = np.cumsum(w)
    p = np.random.rand() # Cumulative probability index, start off random
    idx = 0
    for i in range(target_points):
        while idx < weights.size and not (p >= w[idx] and p < w[idx+1]):
            idx += 1
        idx = idx % weights.size
        choices[i] = order[idx]
        p = (p + 1/target_points) % 1
    uv = np.array(list(np.unravel_index(choices, weights.shape)), dtype=float).T
    if jitter > 0:
        uv += jitter*np.random.randn(uv.shape[0], uv.shape[1])
    return uv

X_GRIDS = {}
def get_nn_weights(vcoords, shape, n_neighbs, device, temperature=20):
    """
    For each pixel/voxel, compute the nearest neighbors in the sites

    Parameters
    ----------
    vcoords: torch.tensor(n_sites, dim)
        Coordinates of voronoi sites
    shape: tuple of (M, N, ..., n_channels)
        Shape of the image we're trying to approximate
    n_neighbs: int
        Number of sites to use as nearest neighbors for each point
    device: str
        Torch device
    temperature: float
        Temperature to use in softmax for neighbor weights
    
    Returns
    -------
    nn_idx: ndarray(M*N*..., n_neighbs)
        Indices of k-nearest neighbors for each pixel/voxel
    weights: torch.tensor(M*N*..., n_neighbs)
        Weights of each neighbor for each pixel/voxel
    """
    import torch
    from scipy.spatial import KDTree
    ## Step 1: Setup grid of pixels
    if not shape in X_GRIDS:
        X = torch.meshgrid(*[torch.linspace(0, 1, n) for n in shape[0:-1]], indexing='ij')
        X = torch.concatenate(tuple([Xi.unsqueeze(-1) for Xi in X]), dim=-1).flatten(0, -2).to(device)
        X_GRIDS[shape] = X # Cache for future iterations since this doesn't change
    X = X_GRIDS[shape]

    ## Step 2: Compute distance from all pixels to landmarks
    tree = KDTree(vcoords.detach().cpu().numpy())
    _, nn_idx = tree.query(X.detach().cpu().numpy(), k=n_neighbs) # Nearest landmark indices to pixels
    if n_neighbs == 1:
        nn_idx = nn_idx[:, None]

    ## Step 3: Based on the landmark locations, compute the weight influence
    ## of each landmark on each pixel using softmax
    weights = torch.zeros(np.prod(shape[0:-1]), n_neighbs).to(device)
    for i in range(n_neighbs):
        dX = vcoords[nn_idx[:, i], :] - X
        weights[:, i] = torch.sum(dX**2, dim=1)
    mn = torch.max(weights, dim=1).values
    mn = mn.view(mn.numel(), 1)
    weights = torch.exp(-temperature*weights/mn)
    weights = weights/torch.sum(weights, dim=1, keepdims=True)

    return nn_idx, weights

def get_avg_colors(I, vcoords, nn_idx):
    """
    Compute the average color of the pixels in each region

    Parameters
    ----------
    I: torch.tensor(M, N, ..., n_channels)
        Image we're trying to approximate
    vcoords: torch.tensor(n_sites, dim)
        Coordinates of voronoi sites
    nn_idx: ndarray(M*N*..., n_neighbs)
        Indices of k-nearest neighbors for each pixel/voxel
    weights: torch.tensor(M*N*..., n_neighbs)

    Returns
    -------
    colors: torch.tensor(n_sites, n_channels)
        Voronoi site colors
    colors_valid: ndarray(n_sites)
        Whether a site had at least one pixel near it
    """
    import torch
    from scipy import sparse
    n_channels = I.shape[-1]
    rows = ((nn_idx[:, 0][:, None])*np.ones((1, n_channels))).flatten()
    cols = (np.ones((nn_idx.shape[0], 1))*(np.arange(n_channels)[None, :])).flatten()
    colors = sparse.coo_matrix((I.cpu().numpy().flatten(), (rows, cols)), shape=(vcoords.shape[0], n_channels)).toarray()
    counts = sparse.coo_matrix((np.ones(nn_idx.shape[0]), (nn_idx[:, 0], np.zeros(nn_idx.shape[0]))), shape=(vcoords.shape[0], 1)).toarray()
    denom = np.array(counts)
    denom[denom == 0] = 1
    colors = torch.from_numpy(colors/denom).to(vcoords)
    return colors, counts > 0

def splat_colors(nn_idx, weights, colors, shape, device):
    """
    Splat the colors from each landmark over the pixels within their influence

    Parameters
    ----------
    nn_idx: ndarray(M*N*..., n_neighbs)
        Indices of k-nearest neighbors for each pixel/voxel
    weights: torch.tensor(M*N*..., n_neighbs)
        Weights of each neighbor for each pixel/voxel
    colors: torch.tensor(n_sites, n_channels)
        Voronoi site colors
    shape: tuple of (M, N, ..., n_channels)
        Shape of the image we're trying to approximate
    device: str
        Torch device

    Returns
    -------
    J: torch.tensor(M, N, ..., n_channels)
        Voronoi image
    """
    import torch
    n_channels = shape[-1]
    n_voxels = nn_idx.shape[0]
    n_neighbs = nn_idx.shape[1]
    J = torch.zeros(n_voxels, n_channels).to(device)
    for i in range(n_neighbs):
        J += colors[nn_idx[:, i], :]*(weights[:, i].view(n_voxels, 1))
    J = J.view(shape)
    return J

def get_voronoi_image(I, device, n_sites, n_neighbs=2, temperature=20, n_iters=50, lr=2e-2, splat_intervals=[], plot_iter_interval=0, verbose=False):
    """
    Compute a set of Voronoi sites best fit to an image

    Parameters
    ----------
    I: ndarray(M, N, n_channels)
        Original image
    device: str
        Torch device
    n_sites: int
        Number of Voronoi sites
    n_neighbs: int
        Number of nearest neighbors to use in the voronoi diagram (default 2)
    temperature: float
        Temperature to use in softmax for neighbor weights (default 20)
    n_iters: int
        Number of iterations of gradient descent
    lr: float
        Learning rate
    splat_intervals: list
        List of (i1, i2) of the intervals of points in which to accumulate Voronoi splatting loss.
        If the list is empty, just do Voronoi splatting once on the whole thing
    plot_iter_interval: int
        If > 0, make a plot showing the gradient descent every time this number of iterations passes
    verbose: bool
        Whether to plot information about the iterations

    Returns
    -------
    J: torch.tensor(M, N, ..., n_channels)
        Voronoi image
    X: ndarray(<= n_sites, dim+n_channels)
        xy.. and rgb.. stacked up in consecutive columns.  Only including samples
        that have valid coordinates and whose centers are closest to at least 
        one point in the original image
    final_cost: float
        Final error of the optimization
    """
    import torch
    if len(splat_intervals) == 0:
        splat_intervals = [(0, n_sites)]
    ## Step 1: Load Images
    if np.issubdtype(I.dtype, np.integer):
        I = I/255
    I_orig = np.array(I, dtype=np.float32)
    I = torch.from_numpy(I).to(device)
    IGray = np.array(I_orig)
    if IGray.shape[2] == 1:
        IGray = IGray[:, :, 0]
    else:
        assert(IGray.shape[2] >= 3) # If it has more than one, it should have at least 3 channels
        IGray = 0.2125*IGray[..., 0] + 0.7154*IGray[..., 1] + 0.0721*IGray[..., 2]
    
    ## Step 2: Compute weights and sample randomly according to them
    # Chop off boundaries of each axis to make sure we don't choose points there, to
    # avoid singularities with logit
    weights = sobel(IGray)
    for i in range(len(weights.shape)):
        weights = np.take(weights, np.arange(1, weights.shape[i]-1), axis=i)
    vcoords = stochastic_universal_sample(weights, n_sites) + 1
    vcoords /= (np.array(IGray.shape)-1)[None, :]
    idx_perm, _ = get_greedy_perm(vcoords)
    vcoords = vcoords[idx_perm, :]
    vcoords = torch.from_numpy(vcoords)
    vcoords = torch.log(vcoords/(1-vcoords)) # Do logit (inverse sigmoid)
    vcoords = vcoords.to(device)
    vcoords = vcoords.requires_grad_()

    ## Step 3: Do the regression!
    optimizer = torch.optim.Adam([vcoords], lr=lr)
    img_costs   = [[] for _ in range(len(splat_intervals))]

    plot_rows = int(np.ceil(len(splat_intervals)/2.0))
    plot_cols = int(np.ceil(2*len(splat_intervals)/plot_rows))
    plot_rows += 1
    print("plot_rows", plot_rows, ", plot_cols", plot_cols)
    if plot_iter_interval > 0:
        plt.figure(figsize=(plot_cols*5, plot_rows*5))
    plot_idx = 0

    for i in range(n_iters):
        tic = time.time()
        optimizer.zero_grad()
        loss = 0
        Js = []
        for k, (i1, i2) in enumerate(splat_intervals):
            vcoordsk = torch.sigmoid(vcoords[i1:i2, :])
            nn_idx, weights = get_nn_weights(vcoordsk, I.shape, n_neighbs, device, temperature)
            colors, _ = get_avg_colors(I, vcoords, nn_idx)
            J = splat_colors(nn_idx, weights, colors, I.shape, device)
            Js.append(J)
            img_cost = torch.sum((I-J)**2)
            loss += img_cost
            img_costs[k].append(img_cost.item())
            """
            from videoutils import save_video
            J = J.detach().cpu().numpy()
            J = np.array(np.floor(J*255), dtype=np.uint8)
            save_video("Iter{}.avi".format(i), J)
            """
        loss.backward()
        optimizer.step()

        ## Make plots showing progress if requested (assuming 2D image)
        if plot_iter_interval > 0 and i % plot_iter_interval == 0:
            M, N = I.shape[0], I.shape[1]
            xydisp = torch.sigmoid(vcoords).detach().cpu().numpy()
            plt.clf()
            for k, (J, (i1, i2)) in enumerate(zip(Js, splat_intervals)):
                plt.subplot(plot_rows, plot_cols, k*2+1)
                plt.imshow(I_orig, aspect='auto')
                plt.axis("off")
                plt.scatter(xydisp[i1:i2, 1]*N, xydisp[i1:i2, 0]*M, s=1)
                plt.title("({}, {})".format(i1, i2))
                
                plt.subplot(plot_rows, plot_cols, k*2+2)
                J = J.detach().cpu().numpy()
                plt.imshow(J, aspect='auto')
                plt.axis("off")
                plt.title("{:.3f}".format(img_costs[k][-1]))

                plt.subplot(plot_rows, plot_cols, 2*plot_cols+k+1)
                plt.plot(img_costs[k])
                plt.xlabel("Iteration")
                if k == 0:
                    plt.ylabel("Loss")
                plt.title("({}, {}), {:.3f}".format(i1, i2, img_costs[k][-1]))
            plt.savefig("Voronoi{}.png".format(plot_idx), bbox_inches='tight')
            plot_idx += 1

        if verbose:
            print("Iteration {}, Loss {:.3f}, Elapsed Time: {:.3f}".format(i, loss.item(), time.time()-tic))
    
    vcoords = torch.sigmoid(vcoords)
    nn_idx, weights = get_nn_weights(vcoords, I.shape, n_neighbs, device, temperature)
    colors, colors_valid = get_avg_colors(I, vcoords, nn_idx)
    J = splat_colors(nn_idx, weights, colors, I.shape, device)
    final_cost = torch.sum((I-J)**2)
    # Stack up vcoords and colors in columns and keep only valid ones
    colors_valid = colors_valid.flatten()
    vcoords_cpu = vcoords.detach().cpu().numpy()
    colors_cpu = colors.detach().cpu().numpy()
    X = np.concatenate((vcoords_cpu, colors_cpu), axis=1)
    valid_idxs = np.prod(~np.isnan(X), axis=1)
    valid_idxs = np.bitwise_and(valid_idxs, colors_valid)
    X = X[valid_idxs == 1, :]
    return J.detach().cpu().numpy(), X, final_cost.item()
