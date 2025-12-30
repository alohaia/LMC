"""Basic common data operations.

Shared low-level data, Vertices for instance, operations are implemented here.
"""

from typing import Literal, Union

from igraph import Graph
from pandas.plotting import deregister_matplotlib_converters
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, MultiPolygon, shape

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from lmc.types import *


def get_vs(graph: Graph, z: bool = True) -> Vertices:
    """Get coordinates of all vertices in the given Graph object."""
    return np.array(
        (graph.vs["x"], graph.vs["y"], graph.vs["z"])
        if z else (graph.vs["x"], graph.vs["y"])
    ).T


def filter_vs(
    graph: Graph,
    attr: str | tuple[str, ...] = "", attr_not: str | tuple[str, ...] = "",
    vtype: str | tuple[str | float, ...] = "",
    z: bool = True
) -> Vertices:
    """Filter vertices according to a boolean attribute and/or vertex type.

    Args:
        graph: Source Graph object.
        attr: Attribute name(s).
        attr_not: Attribute name(s).
        vtype: Type(s) of target vertices. "" is treated the same as NaN.
        z: Whether to include z coordinate.

    Returns:
        N * 2(x, y) array containing coordinates of filtered vertices.
    """
    vs_mask = np.full((graph.vcount(),), fill_value=True)

    if attr != "" and attr != ():
        attr_tuple = (attr,) if isinstance(attr, str) else attr
        for a in attr_tuple:
            vs_mask &= np.array(graph.vs[a]) == True

    if attr_not != "" and attr_not != ():
        attr_not_tuple = (attr_not,) if isinstance(attr_not, str) else attr_not
        for a_not in attr_not_tuple:
            vs_mask &= np.array(graph.vs[a_not]) == True

    if vtype != "" and vtype != ():
        vtype_tuple = (vtype,) if isinstance(vtype, str) else vtype
        vs_type = np.array(graph.vs["type"])
        vs_mask &= np.isin(vs_type, vtype_tuple) \
            | ((pd.isna(vs_type) | (vs_type == ""))
               & (pd.isna(np.array(vtype_tuple)).any() or "" in vtype_tuple))

    return np.array(
        (graph.vs["x"], graph.vs["y"], graph.vs["z"])
        if z else (graph.vs["x"], graph.vs["y"])
    ).T[vs_mask]


def get_es(graph: Graph, z: bool = True) -> Edges:
    """Get all edges in the given Graph object."""
    vs = get_vs(graph, z=z)
    es_vids = graph.get_edgelist()
    return vs[es_vids]


def filter_es(
    graph: Graph,
    attr: str | tuple[str, ...] = "", attr_not: str | tuple[str, ...] = "",
    etype: str | tuple[str | float, ...] = (),
    z: bool = True
) -> Edges:
    """Filter edges according to a boolean attribute and/or edge type..

    Args:
        graph: Source Graph object.
        attr: Attribute name(s).
        attr_not: Attribute name(s).
        etype: Type(s) of target edges.
            0-PA; 1-PV; 2-DA; 3-AV; 4-C

    Returns:
        N * 2(source, target) * 2(x, y) representing filtered vertices.
    """
    vs = get_vs(graph, z=z)
    es_vids = np.array(graph.get_edgelist())

    es_mask = np.full((es_vids.shape[0],), fill_value=True)

    if attr != "" and attr != ():
        attr_tuple = (attr,) if isinstance(attr, str) else attr
        for a in attr_tuple:
            es_mask &= np.array(graph.es[a]) == True

    if attr_not != "" and attr_not != ():
        attr_not_tuple = (attr_not,) if isinstance(attr_not, str) else attr_not
        for a_not in attr_not_tuple:
            es_mask &= np.array(graph.es[a_not]) != True

    if etype != ():
        etype_tuple = (etype,) if isinstance(etype, int) else etype
        es_type = np.array(graph.es["type"])
        es_mask &= np.isin(es_type, etype_tuple) \
            | (pd.isna(es_type) & pd.isna(np.array(etype_tuple)).any())

    return vs[es_vids[es_mask]]


def calc_es_length(g: Graph, overwrite: bool = True) -> NDArray[np.float64]:
    """Calculate and add edges' lengthes based on coordinates of vertices.

    Args:
        g: The graph to be modified.
        overwrite: Whether to overwrite existing "length" attribute of `g.vs`.
    """
    if ("length" in g.es.attributes() and not overwrite):
        return np.empty((0,))

    es = g.get_edge_dataframe().loc[:, ["source", "target"]]
    vs = g.get_vertex_dataframe().loc[:, ["x", "y"]]
    # Graph internally use vs.index in es["source"] and es["target"]
    vs["vindex"] = vs.index

    es = es.merge(
        vs.rename(columns={"vindex": "source", "x": "x_src", "y": "y_src"}),
        how="left", on="source",
    ).merge(
        vs.rename(columns={"vindex": "target", "x": "x_tgt", "y": "y_tgt"}),
        how="left", on="target",
    )

    return np.sqrt((es.x_src - es.x_tgt) ** 2 + (es.y_src - es.y_tgt) ** 2)


def voronoi_tessalation(
    points: Vertices, ghost_points: Vertices,
    points_annot: Union[str, list[str], tuple[str], NDArray[np.str_]]
) -> VoronoiExt:
    """Create from given points and ghost points.

    Args:
        points: Given points.
        ghost_points: Ghost Points as boundary.
        points_annot: Annotation for `points`.

    Returns:
        A Voronoi object whose `point_type` attribute is set to
            ```
            [points_annot] * n(points) + ["Ghost point"] * n(ghost_points)
            ```
    """
    annot = np.empty((0,), dtype=np.str_)
    if isinstance(points_annot, str):
        annot = np.array([points_annot] * points.shape[0]
                         + ["Ghost point"] * ghost_points.shape[0])
    elif (isinstance(points_annot, (list, tuple, np.ndarray))
          and len(points_annot) == points.shape[0]):
        annot = np.array(points_annot
                         + ["Ghost point"] * ghost_points.shape[0])
    else:
        raise ValueError("Unexpected type or length of points_annot.")

    return VoronoiExt(np.vstack((points, ghost_points)), point_annot=annot)


def sample_new_DA_point(
    vor: VoronoiExt,
    vor_initial: VoronoiExt,
    min_distance: float,
    target_density: float,
    nr_tries_max: int = 10
) -> tuple[bool, Vertex]:
    """Sample a new point in Voronoi tessalation.

    Steps:
        - Set coordinate range for new point by `vor_initial` (using vor leads
          to uncontrolled expansion).
        - Uniformly sample a new point and find its valid corresponding
          region point in the initial Voronoi geometry. Up to `nr_tries_max`
          times to find a valid region point.
        - Find its valid corresponding region point in the new Voronoi
          geometry.
        - Check density of corresponding region and closest distance from the
          new point to existing points on the current Voronoi geometry.
        - Return whether succeed in find the new point and its coordinate.

    Args:
        vor: VoronoiExt object to which the point is to be added.
        is_ghost_point: Which of `vor.points` are ghost points.
        vor_initial: The initial VoronoiExt object without any additional points.
        is_ghost_pt_initial: Which of `vor_initial.points` are ghost points.
        min_distance: Min distance from new.
        target_density: Target point density (unit: number of points/mm^2).
        nr_tries_max: Max number of tries to sample a new point.

    Returns:
        - Whether a valid new point was sucessfully sampled.
        - Coordinates of new point. A 0*2 empty (not initialized) array will be
          returned if failed.
    """
    x_min = np.min(vor_initial.points[:, 0])
    x_max = np.max(vor_initial.points[:, 0])
    y_min = np.min(vor_initial.points[:, 1])
    y_max = np.max(vor_initial.points[:, 1])

    new_pt = np.empty((2,))

    nr_tries = 0
    is_valid_point = False
    while (not is_valid_point) and nr_tries < nr_tries_max:
        nr_tries += 1

        new_pt = np.array(
            [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]
        )

        # index to the initial region_point corresponds to xy_new_point
        _, i_vor_region_pt_initial = find_voronoi_region_contains_pt(
            vor_initial, new_pt
        )
        # not in range which is allowed from initial geometry
        if i_vor_region_pt_initial < 0:
            continue # sample new point
        else:
            _, it_vor_region_pt = find_voronoi_region_contains_pt(
                vor, new_pt
            )
            assert it_vor_region_pt >= 0

            _, areas_vor_point_regions = get_areas_voronoi_polygons(vor)
            density = 1.e6 / areas_vor_point_regions[it_vor_region_pt]
            closest_distance = get_closest_distance(
                vor.points[vor.point_annot == "DA root"], new_pt
            )
            if (density < target_density and closest_distance > min_distance):
                is_valid_point = True
            else:
                continue

    if is_valid_point:
        print("New DA coords: ", new_pt,
              "\tNr. of tries:", nr_tries,
              "\tIs valid DA:", is_valid_point)
    else:
        print(f"No valid DA found after {nr_tries} tries...")

    return is_valid_point, new_pt.reshape(2)


def find_voronoi_region_contains_pt(
    vor: VoronoiExt, point: Vertex
) -> tuple[np.intp, np.intp]:
    """Get the region and the corresponding point id of a point.

    - Iterate all points and corresponding regions, skip
      - "ghost regions",
      - empty regions (appears when two points, useally ghost points, has the
        same coordinates),
      - open region.
      Then check if the given point is in the region.
    - Return Index of the region and index of the region point.

    Args:
        vor: VoronoiExt object to be based on.
        point: The given point.

    Returns:
        - Index of the region containing the seed_point.
        - Index of the point corresponding to the region containing the given
          point.

        `tuple(-1, -1)` indicates that the given point is not in a valid region.
    """
    for i_point in np.arange(np.size(vor.point_region)):
        # index of region (in vor.regions) corresponding to vor.points[i_point]
        i_region = vor.point_region[i_point]

        # not valid region
        if vor.point_annot[i_point] == "Ghost point" or \
            len(vor.regions[i_region]) < 3 or \
            (-1 in vor.regions[i_region]):
            continue
        else:
            if is_in_voronoi_region(vor, i_region, point):
                return i_region, i_point
            else:
                continue

    return np.intp(-1), np.intp(-1)


def get_areas_voronoi_polygons(
    vor: VoronoiExt
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Get areas of each polygons in a `VoronoiExt` object.

    Args:
        vor: A `VoronoiExt` object containing multiple polygons.

    Returns:
        Two array of areas in same length and different order.

        - Areas of each region in `vor.regions`.
        - Areas of each region in `vor.point_region`.

        -1 means open geometry.
    """
    # surface area of each polygon
    area_vor_regions: NDArray[np.float64] = np.empty((0, ))

    # find area of each voronoi polygon
    # i (list[int]) is a list of indices to vertices of each region
    for i in vor.regions:
        if len(i) > 2 and (-1 not in i):  # -1 or only 2 vertices: not closed
            area_vor_regions = np.append(area_vor_regions,
                                         polygon_area(vor.vertices[i,]))
        else:
            area_vor_regions = np.append(area_vor_regions, -1)

    nr_points = np.size(vor.point_region)
    area_vor_pt_regions = np.array(
        [area_vor_regions[vor.point_region[i_pt]] for i_pt in range(nr_points)]
        #                 ^^^^^^^^^^^^^^^^^^^^^^
        #                 region index corresponding to vor.points[i_pt]
    )
    return area_vor_regions, area_vor_pt_regions


def get_closest_distance(
        points: Vertices,
        target_point: Vertex
) -> np.float64:
    """Calculate minimal "Euclidean norm" of `points` to `target_point`."""
    return np.min(np.linalg.norm(points - target_point.reshape(-1, 2), axis=1))


def polygon_area(
    coords: Vertices
) -> np.float64:
    """Calculate area of a polygon composed of given vertices.

    Shoelace formula for area of polygon: https://stackoverflow.com/questions/
    24467972/calculate-area-of-polygon-given-x-y-coordinates
    """
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def is_in_voronoi_region(
    vor: VoronoiExt,
    i_region: np.intp,
    point: Vertex
) -> bool:
    """Check if a certain point is within a certain vor region"""
    assert np.size(point) == 2

    current_region_vs = vor.regions[i_region]
    return Point(point).within(Polygon(vor.vertices[current_region_vs]))


def gen_ghost_points(
    points: Vertices,
    radius: float = 800,
    simplify_tolerance: float = 10
) -> Vertices:
    """Generate ghost points as boundary for following Voronoi tessalation.

    Steps:
        - Fuse all circles generate from `points` and `radius`.
        - Simplify boundary.
        - Return the boundary vertices.

    Args:
        points: N*2 array of original points.
        radius: Radius to buffer points.
        simplify_tolerance:
            Tolerance of `Polygon.simplify ` to reduce result points.

    Returns:
        Boundary points as ghost points.
    """
    assert points.shape[1] == 2

    print("Based on the union of circles with midpoint points and radius R.\n"
        f"\tRadius = {radius}\n"
        f"\tResolution Simplifying tolerance = {simplify_tolerance}\n")

    # Generate circles
    circ_list: list[Polygon] = [
        Point(x, y).buffer(radius) for x, y in points
    ]
    # Union all circles
    union_polygon = unary_union(circ_list)
    # Simplify union_polygon to reduce boundary points
    union_polygon = union_polygon.simplify(simplify_tolerance,
                                           preserve_topology=True)

    boundary_points = []
    if isinstance(union_polygon, Polygon):
        boundary_points = list(union_polygon.exterior.coords)
    elif isinstance(union_polygon, MultiPolygon):
        largest_poly = max(union_polygon.geoms, key=lambda p: p.area)
        boundary_points = list(largest_poly.exterior.coords)
    else:
        raise ValueError("unknown union polygon type")

    return np.array(boundary_points).reshape(-1, 2)


def sample_new_AV_point(
    all_segments,
    lengths_segments,
    xy_DA_roots,
    xy_AV_roots,
    min_dist_2_DA,
    min_dist_2_AV,
    nr_tries_max = 10
) -> tuple[bool, Vertex]:

    # Individual and cumulative sum of all segment lengths, normalized with
    # total length of all segments
    rel_lengths_segments = lengths_segments / np.sum(lengths_segments)
    rel_lengths_cumulati = \
        np.cumsum(lengths_segments) / np.sum(lengths_segments)

    is_valid_point = False
    xy_new_AV_candidate = np.empty((2,))
    it = 0

    while (not is_valid_point) and it < nr_tries_max:

        it += 1

        # sample new location with uniform distribution
        sample_loc = np.random.rand()  # uniformly distributed random number
        sample_segment_index = \
            np.where(rel_lengths_cumulati > sample_loc)[0][0]

        rel_loc_up = rel_lengths_cumulati[sample_segment_index]
        rel_loc_delta = rel_lengths_segments[sample_segment_index]
        rel_loc_low = rel_loc_up - rel_loc_delta

        xy_new_AV_candidate = all_segments[sample_segment_index][0, :] \
            + (sample_loc - rel_loc_low) / rel_loc_delta \
            * (all_segments[sample_segment_index][1, :] \
            - all_segments[sample_segment_index][0, :])

        if np.size(xy_DA_roots,axis=0) > 0:
            closest_distance_2_DAs = get_closest_distance(xy_DA_roots,
                                                          xy_new_AV_candidate)
        else:
            # set a very large value, such that always True
            closest_distance_2_DAs = np.sum(lengths_segments) * 100

        if np.size(xy_AV_roots,axis=0) > 0:
            closest_distance_2_AVs = get_closest_distance(xy_AV_roots,
                                                          xy_new_AV_candidate)
        else:
            # set a very large value, such that always True
            closest_distance_2_AVs = np.sum(lengths_segments) * 100

        if (closest_distance_2_DAs > min_dist_2_DA
                and closest_distance_2_AVs > min_dist_2_AV):
            # accept
            is_valid_point = True
        else:
            continue

    if is_valid_point:
        print("New AV coords: ", xy_new_AV_candidate,
                "\tNr. of tries:", it,
                "\tIs valid DA:", is_valid_point)
    else:
        print("No valid AV found after", it, "tries...")

    return is_valid_point, xy_new_AV_candidate.reshape(2)

def create_stacked_hex_network(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
    dist_between_nodes: float,
    l_vessel: float, d_vessel: float,
    perturb_vs_frac: float
) -> Graph:
    # pre-process arguments {{{
    dx_box = x_max - x_min
    dy_box = y_max - y_min
    dz_box = z_max - z_min

    dz_layer = dist_between_nodes
    n_layer = int(np.ceil(dz_box / dz_layer))

    l_hex_edge = 2 * dist_between_nodes
    # }}}

    # generate a single honeycomb layer {{{
    dx_cell = 1.5 * l_hex_edge
    dy_cell = np.sqrt(3.) * l_hex_edge

    n_cells_x = np.int_(np.ceil(dx_box / dx_cell))
    n_cells_y = np.int_(np.ceil(dy_box / dy_cell))

    print('Nr of hexagon in x', n_cells_x)
    print('Nr of hexagon in y', n_cells_y)

    # minimal basic vector
    # also y, x (see below)
    _vec = np.array((l_hex_edge / 2, l_hex_edge * np.sqrt(3) / 2))
    # [0: p, 1: q], (x, y)
    vs_pq0 = (_vec * (0, 1), _vec * (1, 0))
    # x_cell, (dx, dy)
    offsets_x = np.array([
        _vec * (3 * ic_x, 0 if ic_x % 2 == 0 else 1)
        for ic_x in range(n_cells_x)
    ])
    # y_cell, x_cell, (dx, dy)
    # notice: MUST be y, x to make reshape works properly
    offsets_xy = offsets_x[np.newaxis, ] \
        + (
            # y_cell, (ddx, ddy)
            (_vec * (0, 2))[np.newaxis, :] \
                * np.arange(n_cells_y)[:, np.newaxis]
        )[:, np.newaxis, ]
    # y_cell, x_cell, [0: p, 1: q], (dx, dy)
    vs_pq = np.array(vs_pq0) + offsets_xy[:, :, np.newaxis]
    vs_pq = vs_pq.reshape(-1, 2)

    # n, (x, y)
    vs_a = _vec * (3, 0) \
        + _vec * (6, 0) * np.arange(n_cells_x // 2)[:, np.newaxis]
    vs_b = _vec * (6, 2 * n_cells_y + 1) \
        + _vec * (6, 0) * np.arange(n_cells_x // 2)[:, np.newaxis]
    vs_c = _vec * (1, 2 * n_cells_y) \
        + _vec * (3, 0) * np.arange(n_cells_x)[:, np.newaxis]
    vs_c[np.arange(n_cells_x // 2) * 2 + 1, ] += _vec * (0, 1)
    # n, [0: d1, 1: d2], (x, y)
    if n_cells_x // 2 == 0:
        vs_d1d2 = [_vec * (3 * n_cells_x, 1), _vec * (3 * n_cells_x + 1, 2)] \
            + _vec * (0, 2) * np.arange(n_cells_y)[:, np.newaxis, np.newaxis]
    else:
        vs_d1d2 = [_vec * (3 * n_cells_x, 0), _vec * (3 * n_cells_x + 1, 1)] \
            + _vec * (0, 2) * np.arange(n_cells_y)[:, np.newaxis, np.newaxis]

    indices_a = (np.arange(np.size(vs_a) / 2) + np.size(vs_pq) / 2) \
        .astype(np.intp)
    indices_b = (np.arange(np.size(vs_b) / 2) + indices_a[-1] + 1) \
        .astype(np.intp)
    # a special when n_cells_x is odd, add it a B vertices,
    # for convenience of following operations
    if n_cells_x % 2 == 1:
        vs_b = np.vstack((vs_b, _vec * (n_cells_x * 3, n_cells_y * 2)))
        indices_b = np.append(indices_b, indices_b[-1] + 1)

    indices_c = (np.arange(np.size(vs_c) / 2) + indices_b[-1] + 1) \
        .astype(np.intp)
    indices_d1d2 = (np.arange(np.size(vs_d1d2) / 2) + indices_c[-1] + 1) \
        .astype(np.intp).reshape(-1, 2)


    n_vs_x = 2 * n_cells_x
    es_123 = np.zeros((n_cells_y, n_cells_x, 3, 2), dtype=np.intp)
    es_a = np.empty((0, 2), dtype=np.intp)
    es_b = np.empty((0, 2), dtype=np.intp)
    es_c = np.empty((0, 2), dtype=np.intp)
    for ic_y in range(n_cells_y):
        for ic_x in range(n_cells_x):
            # initial index of this cell to vs_pq
            _i0 = 2 * (ic_x + n_cells_x * ic_y)

            if ic_y != n_cells_y - 1:
                e1 = (_i0 + n_vs_x + 1, _i0)
            else:
                e1 = (indices_c[ic_x], _i0)

            e2 = (_i0, _i0 + 1)

            if ic_x == n_cells_x - 1:
                e3 = (_i0 + 1, indices_d1d2[ic_y, 0])
            elif ic_x % 2 == 1:
                e3 = (_i0 + 1, _i0 + 2)
            elif ic_y == 0:
                e3 = (_i0 + 1, indices_a[ic_x // 2])
            else:
                e3 = (_i0 + 1, _i0 - n_vs_x + 2)

            es_123[ic_y, ic_x, ] = np.array((e1, e2, e3))

            # edges of type a
            if ic_y == 0 and ic_x % 2 == 0 and ic_x != n_cells_x - 1:
                es_a = np.vstack((es_a, (indices_a[ic_x // 2], _i0 + 3)))
            # edges of type b
            if (ic_y == n_cells_y - 1 and ic_y % 2 == 1
                    and ic_x != n_cells_x - 1):
                es_b = np.vstack((es_b, (indices_b[ic_x // 2],
                                         indices_c[ic_x + 1])))
            # edges of type c
            if ic_y == n_cells_y - 1:
                if n_cells_x % 2 == 1 and ic_x == n_cells_x - 1:
                    # connect to special B point
                    es_c = np.vstack((es_c, (indices_c[ic_x], indices_b[-1])))
                else:
                    if ic_x % 2 == 0:
                        es_c = np.vstack((es_c, (indices_c[ic_x], _i0 + 2)))
                    else:
                        es_c = np.vstack((es_c, (indices_c[ic_x],
                                                indices_b[ic_x // 2])))

    # edges of type d1 and d2
    indices_d1_d2_lastb = np.append(indices_d1d2.reshape(-1), indices_b[-1])
    es_d1_d2 = np.array((indices_d1_d2_lastb, np.roll(indices_d1_d2_lastb, -1))).T[:-1]

    es = np.vstack((es_123.reshape(-1, 2), es_a, es_b, es_c, es_d1_d2))
    vs = np.vstack((vs_pq, vs_a, vs_b, vs_c, vs_d1d2.reshape(-1, 2)))
    vtypes = ["p", "q"] * (n_cells_x * n_cells_y) \
        + ["a"] * vs_a.shape[0] \
        + ["b"] * vs_b.shape[0] \
        + ["c"] * vs_c.shape[0] \
        + ["d1", "d2"] * n_cells_y
    etypes = ["e1", "e2", "e3"] * (n_cells_x * n_cells_y) \
        + ["a"] * es_a.shape[0] \
        + ["b"] * es_b.shape[0] \
        + ["c"] * es_c.shape[0] \
        + ["d1", "d2"] * n_cells_y
    # }}}

    # from matplotlib import pyplot as plt
    # from matplotlib.collections import LineCollection
    #
    # fig, ax = plt.subplots(1, 1, figsize=(8,8))
    #
    # vertex_type_to_color = {
    #     "p": "black", "q": "black",
    #     "a": "red",
    #     "b": "green",
    #     "c": "blue",
    #     "d1": "yellow", "d2": "yellow",
    #     "split node": "gray"
    # }
    # vertex_colors = [vertex_type_to_color[t] for t in vtypes]
    # ax.scatter(vs[:, 0], vs[:, 1], c=vertex_colors)
    #
    # edge_type_to_color = {
    #     "e1": "black", "e2": "black", "e3": "black",
    #     "a": "red",
    #     "b": "green",
    #     "c": "blue",
    #     "d1": "yellow", "d2": "yellow",
    #     "connect edge": "gray"
    # }
    # edge_colors = [edge_type_to_color[t] for t in etypes]
    # ax.add_collection(LineCollection(
    #     vs[es].tolist(),
    #     colors=edge_colors,
    #     linewidths=1.5,
    #     alpha=0.8
    # ))
    #
    # ax.set_aspect("equal", adjustable="box")
    # ax.invert_yaxis()
    # fig.show()
    #
    # breakpoint()

    # split edges {{{
    split_nodes = np.sum(vs[es], axis=1) / 2
    n_split_nodes = split_nodes.shape[0]
    vid_split_nodes = np.arange(split_nodes.shape[0]) + vs.shape[0]
    es_split_1 = es.copy()
    es_split_2 = es.copy()
    es_split_1[:, 1] = vid_split_nodes
    es_split_2[:, 0] = vid_split_nodes

    vs = np.vstack((vs, split_nodes))
    es = np.vstack((es_split_1, es_split_2))
    vtypes = vtypes + ["split node"] * split_nodes.shape[0]
    etypes = [x for x in etypes for _ in range(2)]
    # }}}

    # stack layers {{{
    vs_df = pd.DataFrame({
        "x": np.tile(vs[:, 0], n_layer), "y": np.tile(vs[:, 1], n_layer),
        "z": (np.ones(vs.shape[0]) * (np.arange(n_layer) + 1)[:, np.newaxis]) \
            .ravel() * dz_layer,
        "type": np.tile(vtypes, n_layer)
    })
    es_vid_step = (np.arange(n_layer) * vs.shape[0])[:, np.newaxis]
    es_df = pd.DataFrame({
        "source": (es[:, 0] + es_vid_step).ravel(),
        "target": (es[:, 1] + es_vid_step).ravel(),
        "type": np.tile(etypes, n_layer),
        "diameter": d_vessel,
        "length": l_vessel
    })
    # }}}

    # g = Graph.DataFrame(edges=es_df, vertices=vs_df)
    #
    # from lmc import viz
    # from lmc.core import io
    #
    # fig, _ = viz.plot_caps(g)
    # fig.show()
    #
    # io.write_vtk(g, "test.vtk")
    #
    # breakpoint()

    # add connection edges {{{
    connect_nodes_1 = np.random.choice(vid_split_nodes, n_split_nodes // 2,
                                          replace=False)
    connect_nodes_2 = vid_split_nodes[~np.isin(vid_split_nodes,
                                               connect_nodes_1)]
    n_vs_layer = vs.shape[0]
    es_connect_list = []
    for i_layer in np.arange(n_layer - 1) + 1:
        connect_nodes = \
            connect_nodes_1 if i_layer % 2 == 1 else connect_nodes_2
        es_connect_list.append(pd.DataFrame({
            "source": connect_nodes + n_vs_layer * i_layer,
            "target": connect_nodes + n_vs_layer * (i_layer - 1),
            "type": "connect edge",
            "diameter": d_vessel,
            "length": l_vessel
        }))

    es_df = pd.concat((es_df, pd.concat(es_connect_list)))
    # }}}

    # purturb vertices and tranlsate by the way {{{
    min_coords = {"x": x_min, "y": y_min, "z": z_min}
    nr_vs = vs_df.shape[0]
    for ax in ("x", "y", "z"):
        vs_df[ax] += (np.random.random(nr_vs) - 0.5) * dist_between_nodes \
            * perturb_vs_frac + min_coords[ax]
    # }}}

    g = Graph.DataFrame(edges=es_df, vertices=vs_df)

    # from lmc import viz
    # from lmc.core import io
    #
    # fig, _ = viz.plot_caps(g)
    # fig.show()
    #
    # io.write_vtk(g, "test.vtk")
    #
    # breakpoint()

    return g


