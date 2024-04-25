import cv2
import numpy as np


def load_video(filename, make_rgb=False, ycrcb=False, verbose=False, pow2=False):
    """
    Wraps around OpenCV to load a video frame by frame
    
    Parameters
    ----------
    filename: string
        Path to the video file I want to load
    make_rgb: bool
        If True, frames are in RGB order
        If False (default), frames are in BGR order
    ycrcb: bool
        If True, convert to ycrcb
    verbose: bool
        If True, print a . for every frame as it's loaded
    pow2: bool
        If True, zeropad video to nearest power of 2
    
    Returns
    -------
    ndarray(nrows, ncols, nframes, 3)
        A 4D array for the color video
    
    """
    cap = cv2.VideoCapture(filename)
    if (cap.isOpened()== False):
        print("Error opening file " + filename)
        return
    frames = []
    ret = True
    while ret and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if make_rgb:
                # Change to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif ycrcb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            frames.append(frame)
            if verbose:
                print('.', end='')
    frames = np.array(frames)
    # Put dimensions as 0:row, 1:col, 2: color, 3: frame num
    frames = np.moveaxis(frames, (0, 1, 2, 3), (3, 0, 1, 2))
    if verbose:
        print("")
    cap.release()
    if pow2:
        n_rows = int(2**np.ceil(np.log2(frames.shape[0])))
        n_cols = int(2**np.ceil(np.log2(frames.shape[1])))
        frames2 = np.zeros((n_rows, n_cols, frames.shape[2], frames.shape[3]), dtype=frames.dtype)
        dr = int((n_rows - frames.shape[0])/2)
        dc = int((n_cols - frames.shape[1])/2)
        frames2[dr:frames.shape[0]+dr, dc:frames.shape[1]+dc, :, :] = frames
        frames = frames2
    return np.moveaxis(frames, 2, 3)

def save_video(filename, frames, fps=30, is_rgb=False, is_ycrcb=False):
    """
    Wraps around OpenCV to save a sequence of frames to a video file
    
    Parameters
    ----------
    filename: string
        Path to which to write video
    frames: ndarray(nrows, ncols, nframes, 3)
        A 4D array for the color video
    fps: int
        Frames per second of output video (default 30)
    """
    result = cv2.VideoWriter(filename, 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             fps, (frames.shape[1], frames.shape[0]))
    for i in range(frames.shape[2]):
        frame = frames[:, :, i, :]
        if is_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif is_ycrcb:
            frame = cv2.cvtColor(frame, cv2.COLOR_YCrCb2BGR)
        result.write(frame)
    result.release()



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    frames = load_video("../data/videos/pullups.avi")
    save_video("out.avi", frames)