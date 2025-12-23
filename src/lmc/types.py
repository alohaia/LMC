import numpy as np
from numpy.typing import NDArray
from typing import Optional

from scipy.spatial import Voronoi

# shape(2 | 3): vertex(x, y, ?z)
Vertex = NDArray[np.float64]
# shape(N, 2 | 3): i_vertex, vertex(x, y, ?z)
Vertices = NDArray[np.float64]

# shape(N, 2 | 3, 2 | 3): i_edge, vertex_a(x, y, ?z), vertex_b(x, y, ?z)
Edges = NDArray[np.float64]

# list[shape(N, 2 | 3)]
Regions = list[NDArray[np.float64]]


class VoronoiExt(Voronoi):
    point_annot: NDArray[np.str_]

    def __init__(self, points, point_annot: Optional[NDArray[np.str_]],
                 *args, **kwargs):
        super().__init__(points, *args, **kwargs)

        if point_annot is not None:
            self.point_annot = np.array(point_annot)
        else:
            self.point_annot = np.empty((0,), dtype=np.str_)


