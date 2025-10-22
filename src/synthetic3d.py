import numpy as np
import matplotlib.pyplot as plt
import json


class FakeScanner3D(object):
    def __init__(self, filename):
        """
        Load in a .json file generated from ggslac which
        contains all of the depth images and camera information
        Parameters
        ----------
        filename: string
            Path to JSON scan file
        near: float
            Near dist to filter out the boundary between
            the object and the background due to antialiasing
        """
        print("Loading...")
        data= json.load(open(filename, "r"))
        print("Finished loading")
        self.W = data['width']
        self.H = data['height']
        self.fovx = data['fovx']
        self.fovy = data['fovy']
        self.far = data['far']
        self.allDepth = data['allDepth']
        self.cameras = data['cameras']
        self.num_scans = len(self.allDepth)
        self.unpack_scans()

    def unpack_scans(self):
        """
        Reshape the depth and normal scans and unpack the
        information from the [0, 255] RGB channels
        """
        W = self.W
        H = self.H
        allDepth = []
        print(f"Unpacking {self.num_scans} scans...")
        for i in range(self.num_scans):
            # Extract depth scan
            depth = self.allDepth[i]
            depth = np.reshape(depth, (H, W, 4))
            depth = np.array(depth[:, :, 0:2], dtype=float)/255
            depth = depth[:, :, 0] + depth[:, :, 1]/256
            depth = self.far*depth*(256*256)/(256*256-1)
            depth[depth == 0] = np.inf
            print("depth.shape", depth.shape)
            for key in self.cameras[i].keys():
                self.cameras[i][key] = np.array(self.cameras[i][key])
            self.cameras[i]['fovx'] = self.fovx
            self.cameras[i]['fovy'] = self.fovy
            allDepth.append(depth)
        self.allDepth = allDepth
        print("Finished unpacking")

    def get_scan(self, i):
        """
        Get a particular scan in the list of scans
        Parameters
        ----------
        i: int
            Frame index of the scan
        
        Returns
        -------
        {
            'depth': ndarray(M, N)
                The depth scan,
            'camera': {
                'pos': ndarray(3)
                    Position of the camera,
                'up': ndarray(3)
                    Up vector of the camera,
                'right': ndarray(3)
                    Right vector of the camera,
                'fovx': float
                    Field of view in x,
                'fovy': float
                    Field of view in y
            }
        }
        """
        if i >= self.num_scans:
            raise Exception("Requested scan beyond range")
        return {'depth':self.allDepth[i], 'camera':self.cameras[i]}
    
    def get_scan_xyz(self, i, return_structured=False):
        """
        Return a 3D point cloud of all scanned points from the
        point of view of the camera

        Parameters
        ----------
        i: int
            Frame index of the scan
        return_structured: boolean
            If true, return a stack of XYZ with inf values indicating
            missing data so that neighbors can be determined easily.
            If false, return a Nx3 point cloud with only the scanned points

        Returns
        -------
        ndarray(N, 3): The point cloud
        """
        res = self.get_scan(i)
        cam = res['camera']
        Z = res['depth']
        X = np.linspace(-1, 1, Z.shape[1])
        Y = np.linspace(1, -1, Z.shape[0])
        X, Y = np.meshgrid(X, Y)
        X = Z*X*np.tan(cam['fovx']/2)
        Y = Z*Y*np.tan(cam['fovy']/2)
        Z = -Z
        ret = np.array([])
        if not return_structured:
            X = X[np.isinf(Z) == 0]
            Y = Y[np.isinf(Z) == 0]
            Z = Z[np.isinf(Z) == 0]
            ret = np.array([X, Y, Z]).T
        else:
            ret = np.stack((X, Y, Z), axis=-1)
        return ret

    def get_scan_xyz_world(self, i):
        """
        Return a 3D point cloud of all scanned points in world
        coordinates

        Parameters
        ----------
        i: int
            Frame index of the scan

        Returns
        -------
        ndarray(N, 3): The point cloud
        """
        res = self.get_scan(i)
        cam = res['camera']
        depth = res['depth']
        up = cam['up']
        right = cam['right']
        towards = np.cross(up, right)
        x = np.linspace(-1, 1, depth.shape[1])
        y = np.linspace(1, -1, depth.shape[0])
        x, y = np.meshgrid(x, y)
        X = depth[:, :, None]*towards[None, None, :]
        X += np.tan(cam['fovx']/2)*x[:, :, None]*depth[:, :, None]*right[None, None, :]
        X += np.tan(cam['fovy']/2)*y[:, :, None]*depth[:, :, None]*up[None, None, :]
        X = np.reshape(X, (X.shape[0]*X.shape[1], 3))
        X = X[np.isfinite(X[:, 0]), :] + cam['pos']
        return X
    
    def save_scan_frames_json(self, filename):
        """
        Make a video of point cloud scans from the point of view of
        the camera for plotting in the mesh video viewer

        Parameters
        ----------
        filename: string
            Path to which to save the file
        """
        frames = []
        for i in range(len(self.allDepth)):
            VPos = self.get_scan_xyz(i)
            frames.append(dict(VPos=VPos.tolist(), ITris=[]))
        json.dump(frames, open(filename, "w"))

    def save_scan_frames_h5(self, filename):
        """
        Save all of the point clouds as an h5 file

        Parameters
        ----------
        filename: string
            Path to which to save the file
        """
        from deepdish import io
        frames = []
        for i in range(len(self.allDepth)):
            VPos = self.get_scan_xyz(i)
            frames.append(VPos)
        io.save(filename, {"frames":frames})
    
    def save_scan_frames_alldata_h5(self, filename, infvalue=0):
        """
        Save all of the data, even if it's inf
        
        Parameters
        ----------
        filename: string
            Path to which to save the file
        infvalue: float
            Value with which to replace all infs in each frame
        """
        N = len(self.allDepth)
        frames = np.zeros((self.H, self.W, 3, N))
        for i in range(N):
            res = self.get_scan(i)
            cam = res['camera']
            Z = np.array(res['depth'])
            Z[np.isinf(Z)] = infvalue
            X = np.linspace(-1, 1, Z.shape[1])
            Y = np.linspace(1, -1, Z.shape[0])
            X, Y = np.meshgrid(X, Y)
            X = Z*X*np.tan(cam['fovx']/2)
            Y = Z*Y*np.tan(cam['fovy']/2)
            for k, W in enumerate([X, Y, Z]):
                frames[:, :, k, i] = W
        from deepdish import io
        io.save(filename, {"frames":frames})


    def plot_scans(self, prefix=""):
        """
        Make a video of all of the scans
        """
        vmin = np.inf
        vmax = 0
        for depth in self.allDepth:
            vmin = min(vmin, np.min(depth[np.isfinite(depth)]))
            vmax = max(vmax, np.max(depth[np.isfinite(depth)]))
        print("vmin = %.3g"%vmin)
        print("vmax = %.3g"%vmax)
        fac = 1
        plt.figure(figsize=(fac*5, fac*5))
        for i, depth in enumerate(self.allDepth):
            plt.clf()
            plt.imshow(depth, vmin=vmin, vmax=vmax, cmap='magma_r')
            plt.colorbar()
            plt.title("Depth Scan {}".format(i))
            plt.savefig("{}{}.png".format(prefix, i), bbox_inches='tight')
