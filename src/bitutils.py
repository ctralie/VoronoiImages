import numpy as np
import torch

def get_binary_encoding(Y, bits_per_channel):
    """
    Parameters
    ----------
    Y: torch.tensor(n_batches, time, len(bits_per_channel))
        Tensor to encode, with values in [0, 1]
    bits_per_channel: list of num_channels ints
        Number of bits to use in each channel
    
    Returns
    -------
    torch.tensor(n_batches, time, sum(bits_per_channel))
        Binary encoding of all channels
    """
    total_bits = np.sum(bits_per_channel)
    YBin = torch.zeros(Y.shape[0], Y.shape[1], total_bits, dtype=Y.dtype)
    YBin = YBin.to(Y)
    bit_offset = 0
    for dim, bits in enumerate(bits_per_channel):
        Ydim = (2**bits-1)*Y[:, :, dim]
        place = 2**(bits-1)
        while place > 0:
            bit = 1.0*(Ydim-place > 0)
            YBin[:, :, bit_offset] = bit
            Ydim -= bit*place
            place = place // 2
            bit_offset += 1
    return YBin

def decode_binary(YBin, bits_per_channel, normalize=True):
    """
    Parameters
    ----------
    YBin: torch.tensor(n_batches, time, sum(bits_per_channel))
        Tensor to encode, with values in [0, 1]
    bits_per_channel: list of num_channels ints
        Number of bits to use in each channel
    normalize: bool
        If True, normalize each channel to [0, 1]
    
    Returns
    -------
    torch.tensor(n_batches, time, len(bits_per_channel))
        Binary encoding of all channels
    """
    Y = torch.zeros(YBin.shape[0], YBin.shape[1], len(bits_per_channel), dtype=YBin.dtype)
    Y = Y.to(YBin)
    bit_offset = 0
    for dim, bits in enumerate(bits_per_channel):
        Ydim = (2**(bits-1))*YBin[:, :, bit_offset]
        place = 2**(bits-2)
        bit_offset += 1
        while place >= 1:
            Ydim += place*YBin[:, :, bit_offset]
            place = place // 2
            bit_offset += 1
        if normalize:
            Y[:, :, dim] = Ydim/(2**bits-1)
        else:
            Y[:, :, dim] = Ydim
    return Y

def get_gray_codes(n_bits):
    """
    Compute gray code sequences of a specified size
    
    Parameters
    ----------
    n_bits: int
        Number of bits to use
    
    Returns
    -------
    list of lists of n_bits 1's and 0's for gray codes
    """
    res = []
    if n_bits == 1:
        res = [[0], [1]]
    else:
        g = get_gray_codes(n_bits-1)
        res = [[0]+b for b in g] + [[1]+b for b in g[::-1]]
    return res

def encode_gray_angle(B, n_bits):
    """
    Convert a binary sequence into a gray code sequence represented
    by floats in [0, 2*pi]
    
    Parameters
    ----------
    B: ndarray(N)
        An array of bits
    n_bits: int
        Number of bits in the gray code
    
    Returns
    ndarray(ceil(N/n_bits))
        Gray code encoding
    """
    code = get_gray_codes(n_bits)
    code = {tuple(c):2*np.pi*(i+0.5)/2**n_bits for i, c in enumerate(code)}
    N = int(np.ceil(len(B)/n_bits))
    encoding = np.zeros(N)
    for i in range(N):
        c = B[i*n_bits:(i+1)*n_bits]
        if len(c) < n_bits:
            c = c + [0]*(n_bits-len(c)) # Zeropad
        encoding[i] = code[tuple(c)]
    return encoding
    
def decode_gray_angle(x, n_bits):
    """
    Parameters
    ----------
    x: ndarray(N)
        Gray code angle sequence
    n_bits: int
        Number of bits in the gray code
    
    Returns
    -------
    ndarray(N*n_bits)
        Decoded bits
    """
    code = np.array(get_gray_codes(n_bits), dtype=int)
    idx = np.array(np.floor(x*(2**n_bits)/(2*np.pi)), dtype=int)
    idx = idx % (2**n_bits)
    return code[idx].flatten()

def decode_viterbi_bits(y, n_bits, berr, sigma=25):
    """
    Viterbi code a sequence of bits assuming continuity from one state to the next

    Parameters
    ----------
    y: ndarray(T)
        State observations in range[0, 1]
    n_bits: int
        Number of bits used to encode each state
    berr: float
        Assumed bit error rate
    sigma: float
        Penalty for jumping
    
    Returns
    -------
    ndarray(T): Cleaned observations
    """
    K = 2**n_bits
    y = np.array(y*(K-1), dtype=int)
    states = np.zeros((K, n_bits))
    for i in range(K):
        bits = [int(x) for x in bin(i)[2::]]
        states[i, n_bits-len(bits)::] = bits
    T = len(y)
    L = np.zeros((K, T))
    B = np.zeros((K, T), dtype=int)
    pix = np.linspace(0, 1, K)
    csm = -sigma*np.abs(pix[:, None] - pix[None, :])
    LLast = np.zeros(K)
    for t in range(T):
        # Transition probability
        state = states[int(y[t])]
        diff = np.sum((state[None, :] + states) % 2, axis=1)
        obs_prob = -diff*np.log(berr)
        trans = LLast[:, None] + csm
        B[:, t] = np.argmax(trans, axis=0)
        L[:, t] = np.max(trans, axis=0) + obs_prob
        LLast = L[:, t]
    ## Do backtracing
    t = T-1
    j = np.argmax(L[:, -1])
    z = [j]
    while t > 0:
        j = B[j, t]
        z.append(j)
        t -= 1
    z.reverse()
    z = K-1-np.array(z)
    return z/(K-1)

def viterbi_clean_observations(Y, bits_per_channel, berr, sigma=25):
    """
    Viterbi code a sequence of bits assuming continuity from one state to the next

    Parameters
    ----------
    Y: ndarray(T, 5)
        Original xyrgb coordinates
    bits_per_channel: list of 5 ints
        Bits for x, y, z, r, g, b for each voronoi site
    berr: float
        Assumed bit error rate
    sigma: float
        Penalty for jumping
    
    Returns
    -------
    ndarray(T, 5)
        Cleaned xyrgb coordinates
    """
    Z = np.zeros_like(Y)
    for k, n_bits in enumerate(bits_per_channel):
        print(k, n_bits)
        Z[:, k] = decode_viterbi_bits(Y[:, k], n_bits, berr, sigma)
    return Z

def get_bit_error_normalized(Y, Z, bits_per_channel):
    """
    TODO: There seems to be an inconsistency here compared to the original
    bit errors returned from encode_extract_voronoi_image; errors here are reported higher

    Parameters
    ----------
    Y: ndarray(T, 5)
        Ground truth xyrgb coordinates
    Z: ndarray(T, 5)
        Observed xyrgb coordinates
    bits_per_channel: list of 5 ints
        Bits for x, y, z, r, g, b for each voronoi site
    """
    num = 0
    denom = 0
    for k, n_bits in enumerate(bits_per_channel):
        K = 2**n_bits
        states = np.zeros((K, n_bits))
        for i in range(K):
            bits = [int(x) for x in bin(i)[2::]]
            states[i, n_bits-len(bits)::] = bits
        y = np.array(Y[:, k]*(K-1), dtype=int)
        z = np.array(Z[:, k]*(K-1), dtype=int)
        num += np.sum(np.abs(states[y]-states[z]))
        denom += n_bits*Y.shape[0]
    return num/denom