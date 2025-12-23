"""Basic common data operations.

Shared low-level data, Vertices for instance, operations are implemented here.
"""

from typing import Optional, Union

from igraph import Graph
from shapely.ops import unary_union
from shapely.geometry import Point, Polygon, MultiPolygon

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from lmc.types import *


def get_vs(graph: Graph, z: bool = False) -> Vertices:
    """Get coordinates of all vertices in the given Graph object."""
    return np.array(
        (graph.vs["x"], graph.vs["y"], graph.vs["z"])
        if z else (graph.vs["x"], graph.vs["y"])
    ).T


def filter_vs(
    graph: Graph,
    attr: str | tuple[str, ...] = "", attr_not: str | tuple[str, ...] = "",
    vtype: str | tuple[str | float, ...] = "",
    z: bool = False
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


def get_es(graph: Graph) -> Edges:
    """Get all edges in the given Graph object."""
    vs = np.array((graph.vs["x"], graph.vs["y"])).T
    es_vids = graph.get_edgelist()
    return vs[es_vids]


def filter_es(
    graph: Graph,
    attr: str | tuple[str, ...] = "", attr_not: str | tuple[str, ...] = "",
    etype: tuple[str | float, ...] = ()
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
    vs = np.array((graph.vs["x"], graph.vs["y"])).T
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

    if etype != "" and etype != ():
        etype_tuple = (etype,) if isinstance(etype, str) else etype
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
    rel_lengths_cumulati = np.cumsum(lengths_segments) / np.sum(lengths_segments)

    is_valid_point = False
    xy_new_AV_candidate = np.empty((2,))
    it = 0

    while (not is_valid_point) and it < nr_tries_max:

        it += 1

        # sample new location with uniform distribution
        sample_loc = np.random.rand()  # uniformly distributed random number
        sample_segment_index = np.where(rel_lengths_cumulati > sample_loc)[0][0]

        rel_loc_up = rel_lengths_cumulati[sample_segment_index]
        rel_loc_delta = rel_lengths_segments[sample_segment_index]
        rel_loc_low = rel_loc_up - rel_loc_delta

        xy_new_AV_candidate = all_segments[sample_segment_index][0, :] \
            + (sample_loc - rel_loc_low) / rel_loc_delta \
            * (all_segments[sample_segment_index][1, :] \
            - all_segments[sample_segment_index][0, :])

        if np.size(xy_DA_roots,axis=0) > 0:
            closest_distance_2_DAs = get_closest_distance(xy_DA_roots, xy_new_AV_candidate)
        else:
            # set a very large value, such that always True
            closest_distance_2_DAs = np.sum(lengths_segments) * 100

        if np.size(xy_AV_roots,axis=0) > 0:
            closest_distance_2_AVs = get_closest_distance(xy_AV_roots, xy_new_AV_candidate)
        else:
            # set a very large value, such that always True
            closest_distance_2_AVs = np.sum(lengths_segments) * 100

        if closest_distance_2_DAs > min_dist_2_DA and closest_distance_2_AVs > min_dist_2_AV:
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

