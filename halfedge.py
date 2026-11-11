import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay
import skimage
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from numba import njit
import time

NULL_HALF_EDGE = -1
BOUNDARY_HALF_EDGE = -1
BOUNDARY_FACE = -1
N_GHOST_VERTICES = 4


@njit(cache=True)
def color_helper(
    image_dx: np.ndarray,
    centers: np.ndarray,
    vertices: np.ndarray,
    he_face: np.ndarray,
    he_tip: np.ndarray,
    he_pair: np.ndarray,
) -> np.ndarray:
    height = image_dx.shape[0]
    width = image_dx.shape[1] - 1
    colors = np.zeros((vertices.shape[0], 3))
    counts = np.zeros(vertices.shape[0])
    for he in range(len(he_tip)):
        if he_face[he] == -1 or he_face[he_pair[he]] == -1:
            continue
        a, b = centers[he], centers[he_pair[he]]
        sign = 1
        if b[0] < a[0]:
            sign = -1
            a, b = b, a
        i_lo = max(int(np.ceil(a[0] - 0.5)), 0)
        i_hi = min(int(np.ceil(b[0] - 0.5)), height)
        if i_hi <= i_lo:
            continue
        i = np.arange(i_lo, i_hi)
        y = a[1] + (i + 0.5 - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
        js = np.round(y).astype(int)  # pixels left of the boundary
        js[js < 0] = 0
        js[js > width] = width
        v = he_tip[he]
        colors[v] += sign * np.sum(image_dx[i, js], axis=0)
        counts[v] += sign * js.sum()
    n_vertices = vertices.shape[0] - N_GHOST_VERTICES
    colors = colors[:n_vertices] / counts[:n_vertices, None]
    return colors


class HalfEdgeMesh:
    def __init__(self, vertices: np.ndarray, tris: np.ndarray) -> None:
        """
        Assume that tris are consistently oriented
        """
        self.vertices = vertices
        self.tris = tris
        n_vertices = vertices.shape[0]
        n_tris = tris.shape[0]
        he_index = np.arange(tris.size).reshape(tris.shape)
        he_face = np.arange(n_tris)[:, None] * np.ones((1, 3), dtype=int)
        self.he_face = he_face.flatten()
        self.he_tip = np.roll(tris, -1, axis=1).flatten()
        self.he_next = np.roll(he_index, -1, axis=1).flatten()
        self.he_prev = np.roll(he_index, 1, axis=1).flatten()
        self.he_pair = self._init_pairs(tris, he_index, n_vertices)

        self.vertex_to_he = NULL_HALF_EDGE * np.ones(n_vertices, dtype=int)
        self.vertex_to_he[tris.flatten()] = he_index.flatten()

        self._init_boundary()

    @staticmethod
    def _init_pairs(
        tris: np.ndarray, he_index: np.ndarray, n_vertices: int
    ) -> np.ndarray:
        n_half_edges = he_index.size
        rows = np.array([], dtype=int)
        cols = np.array([], dtype=int)
        indices = np.array([], dtype=int)
        for i in range(3):
            rows = np.concatenate((rows, tris[:, i]))
            cols = np.concatenate((cols, tris[:, (i + 1) % 3]))
            indices = np.concatenate((indices, he_index[:, i]))
        ## NOTE: Need to 1-index the sparse matrix so half edge 0 doesn't disappear
        he_pair_mat_transpose = sparse.coo_matrix(
            (indices + 1, (rows, cols)), shape=(n_vertices, n_vertices)
        ).T.tocsr()
        he_pair = BOUNDARY_HALF_EDGE * np.ones(n_half_edges, int)
        he_pair[indices] = he_pair_mat_transpose[rows, cols] - 1
        return he_pair

    def _init_boundary(self) -> None:
        interior_boundary_he = np.where(self.he_pair == BOUNDARY_HALF_EDGE)[0]
        n_half_edges = len(self.he_pair)
        n_new_he = len(interior_boundary_he)
        self.he_face = np.concatenate((self.he_face, BOUNDARY_FACE * np.ones(n_new_he)))
        exterior_boundary_he = n_half_edges + np.arange(n_new_he)
        self.he_pair = np.concatenate(
            (self.he_pair, interior_boundary_he)
        )  # exterior pair to interior
        self.he_pair[interior_boundary_he] = (
            exterior_boundary_he  # interior pair to exterior
        )

        x = self.he_tip[interior_boundary_he]
        self.vertex_to_he[x] = exterior_boundary_he
        y = self.he_tip[self.he_prev[interior_boundary_he]]
        self.he_next = np.concatenate((self.he_next, self.vertex_to_he[y]))
        self.he_tip = np.concatenate((self.he_tip, y))
        self.he_prev = np.concatenate((self.he_prev, np.zeros(n_new_he, dtype=int)))
        self.he_prev[self.vertex_to_he[y]] = exterior_boundary_he

    def get_vertex_half_edges(self, index: int) -> list[int]:
        hedges = []
        he_start = self.vertex_to_he[index]
        if he_start == NULL_HALF_EDGE:
            return hedges
        he = he_start
        for _ in range(len(self.he_face)):
            hedges.append(he)
            # he = self.he_next[self.he_pair[he]] # Walks clockwise
            he = self.he_pair[self.he_prev[he]]  # Walks counter-clockwise
            if he == he_start:
                break
        return hedges

    def get_vertex_neighbors(self, index: int) -> list[int]:
        hedges = self.get_vertex_half_edges(index)
        return [self.he_tip[he] for he in hedges]


class DynamicVoronoi(HalfEdgeMesh):
    def __init__(self, image: np.ndarray, vertices: np.ndarray) -> None:
        height, width = image.shape[0], image.shape[1]
        r = 10 * max(height, width)
        cy, cx = height / 2, width / 2
        ghosts = np.array(
            [[cy - r, cx - r], [cy - r, cx + r], [cy + r, cx - r], [cy + r, cx + r]]
        )
        vertices = np.vstack((vertices, ghosts))
        tris = Delaunay(vertices).simplices
        self._make_ccw(vertices, tris)
        super().__init__(vertices, tris)
        self.image = image
        self.image_dx = np.concatenate(
            (
                np.zeros((height, 1, image.shape[2]), image.dtype),
                np.cumsum(image, axis=1),
            ),
            axis=1,
        )

    def plot(self) -> None:
        tic = time.time()
        centers = self.circumcenters_and_radii2()[0]
        print("Elapsed time centers:", time.time() - tic)
        tic = time.time()
        polygons = self.get_voronoi_polygons(centers)
        print("Elapsed time polygons:", time.time() - tic)
        tic = time.time()
        colors = self.get_voronoi_colors(centers)
        print("Elapsed time colors:", time.time() - tic)
        for polygon, color in zip(polygons, colors):
            color = np.clip(color, 0, 1)
            poly = Polygon(
                polygon[:, ::-1],
                closed=True,  # automatically closes the shape
                facecolor=color,  # interior color
                edgecolor=None,  # border colour
                linewidth=0,
                alpha=1,  # transparency (0‑1)
            )
            plt.gca().add_patch(poly)
        plt.xlim([0, self.image.shape[1]])
        plt.ylim([self.image.shape[0], 0])

    def get_voronoi_polygons(self, centers: np.ndarray | None) -> list[np.ndarray]:
        if centers is None:
            centers = self.circumcenters_and_radii2()[0]
        polygons = []
        for i in range(len(self.vertices) - N_GHOST_VERTICES):
            hedges = self.get_vertex_half_edges(i)
            hedges = np.array(hedges, dtype=int)
            polygons.append(centers[hedges])
        return polygons

    def get_voronoi_colors(self, centers: np.ndarray | None) -> np.ndarray:
        if centers is None:
            centers = self.circumcenters_and_radii2()[0]

        return color_helper(
            self.image_dx,
            centers,
            self.vertices,
            self.he_face,
            self.he_tip,
            self.he_pair,
        )

    @staticmethod
    def _make_ccw(vertices: np.ndarray, tris: np.ndarray) -> None:
        a = vertices[tris[:, 0]]
        b = vertices[tris[:, 1]]
        c = vertices[tris[:, 2]]
        ab = np.hstack((b - a, np.zeros((a.shape[0], 1))))
        ac = np.hstack((c - a, np.zeros((c.shape[0], 1))))
        n = np.cross(ab, ac)
        violating = np.where(n[:, -1] < 0)[0]
        if violating.size > 0:
            tris[violating] = tris[violating, ::-1]

    def circumcenters_and_radii2(self):
        """Circumcenters and squared circumradii for triangles."""
        he = np.arange(len(self.he_tip))
        p0 = self.vertices[self.he_tip[he]]
        p1 = self.vertices[self.he_tip[self.he_next[he]]]
        p2 = self.vertices[self.he_tip[self.he_prev[he]]]
        ax, ay = p0[:, 0], p0[:, 1]
        bx, by = p1[:, 0], p1[:, 1]
        cx, cy = p2[:, 0], p2[:, 1]

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = (
            (ax**2 + ay**2) * (by - cy)
            + (bx**2 + by**2) * (cy - ay)
            + (cx**2 + cy**2) * (ay - by)
        ) / d
        uy = (
            (ax**2 + ay**2) * (cx - bx)
            + (bx**2 + by**2) * (ax - cx)
            + (cx**2 + cy**2) * (bx - ax)
        ) / d
        centers = np.stack([ux, uy], axis=1)
        r2 = np.sum((centers - p0) ** 2, axis=1)
        return centers, r2


def main() -> None:
    image = skimage.io.imread("layla.png")
    image = image.astype(np.float32) / 255
    n_sites = 1000
    np.random.seed(0)
    vertices = np.random.rand(n_sites, 2)
    vertices[:, 0] *= image.shape[0]
    vertices[:, 1] *= image.shape[1]
    voronoi = DynamicVoronoi(image, vertices)

    for _ in range(4):
        voronoi.plot()
        plt.show()


if __name__ == "__main__":
    main()
