import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay
import skimage
import matplotlib.pyplot as plt

NULL_HALF_EDGE = -1
BOUNDARY_HALF_EDGE = -1
BOUNDARY_FACE = -1

class HalfEdgeMesh:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """
        Assume that faces are consistently oriented
        """
        self.vertices = vertices
        self.faces = faces
        n_vertices = vertices.shape[0]
        n_faces = faces.shape[0]
        he_index = np.arange(faces.size).reshape(faces.shape)
        he_face = np.arange(n_faces)[:, None] * np.ones((1, 3), dtype=int)
        self.he_face = he_face.flatten()
        self.he_tip = np.roll(faces, -1, axis=1).flatten()
        self.he_next = np.roll(he_index, -1, axis=1).flatten()
        self.he_prev = np.roll(he_index, 1, axis=1).flatten()
        self.he_pair = self._init_pairs(faces, he_index, n_vertices)

        self.vertex_to_he = NULL_HALF_EDGE * np.ones(n_vertices, dtype=int)
        self.vertex_to_he[faces.flatten()] = he_index.flatten()

        self._init_boundary()

    @staticmethod
    def _init_pairs(faces: np.ndarray, he_index: np.ndarray, n_vertices: int) -> np.ndarray:
        n_half_edges = he_index.size
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        indices = np.array([], dtype=int)
        for i in range(3):
            rows = np.concatenate((rows, faces[:, i]))
            cols = np.concatenate((cols, faces[:, (i+1) % 3]))
            indices = np.concatenate((indices, he_index[:, i]))
        ## NOTE: Need to 1-index the sparse matrix so half edge 0 doesn't disappear
        he_pair_mat_transpose = sparse.coo_matrix((indices+1, (rows, cols)),
                                         shape=(n_vertices, n_vertices)).T.tocsr()
        he_pair = BOUNDARY_HALF_EDGE * np.ones(n_half_edges, int)
        he_pair[indices] = he_pair_mat_transpose[rows, cols] - 1
        return he_pair

    def _init_boundary(self) -> None:
        interior_boundary_he = np.where(self.he_pair == BOUNDARY_HALF_EDGE)[0]
        n_half_edges = len(self.he_pair)
        n_new_he = len(interior_boundary_he)
        self.he_face = np.concatenate((self.he_face, BOUNDARY_FACE*np.ones(n_new_he)))
        exterior_boundary_he = n_half_edges + np.arange(n_new_he)
        self.he_pair = np.concatenate((self.he_pair, interior_boundary_he)) # exterior pair to interior
        self.he_pair[interior_boundary_he] = exterior_boundary_he # interior pair to exterior

        x = self.he_tip[interior_boundary_he]
        self.vertex_to_he[x] = exterior_boundary_he
        y = self.he_tip[self.he_prev[interior_boundary_he]]
        self.he_next = np.concatenate((self.he_next, self.vertex_to_he[y]))
        self.he_tip = np.concatenate((self.he_tip, y))
        self.he_prev = np.concatenate((self.he_prev, np.zeros(n_new_he, dtype=int)))
        self.he_prev[self.vertex_to_he[y]] = exterior_boundary_he

    def get_vertex_neighbors(self, index: int) -> list[int]:
        neighbors = []
        he_start = self.vertex_to_he[index]
        if he_start == NULL_HALF_EDGE:
            return neighbors
        he = he_start
        for _ in range(len(self.he_face)):
            neighbors.append(self.he_tip[he])
            # he = self.he_next[self.he_pair[he]] # Walks clockwise
            he = self.he_pair[self.he_prev[he]] # Walks counter-clockwise
            if he == he_start:
                break
        return neighbors


class DynamicVoronoi(HalfEdgeMesh):
    def __init__(self, image: np.ndarray, vertices: np.ndarray) -> None:
        super().__init__(vertices, Delaunay(vertices).simplices)
        self.image = image

        plt.imshow(image)
        for he in range(len(self.he_tip)):
            tip = self.he_tip[he]


def main() -> None:
    image = skimage.io.imread("data/images/layla.png")
    n_sites = 1000
    vertices = np.random.rand(n_sites, 2)
    vertices[:, 0] *= image.shape[0]
    vertices[:, 1] *= image.shape[1]
    voronoi = DynamicVoronoi(image, vertices)

if __name__ == '__main__':
    main()