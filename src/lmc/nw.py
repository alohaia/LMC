"""Higher-level network manipulation toolkit.

*Higher-level* means these functions will be used directly in user-created
scripts. Mainly contains functions manipulating network objects (igraph.Graph).
"""

from pathlib import Path
from typing import Literal

from igraph.drawing import graph
from matplotlib.pyplot import axis, flag
import pandas as pd
import numpy as np

from numpy.typing import NDArray
from pandas.core.internals.blocks import shift
from shapely import polygons
from lmc.types import *

from igraph import Graph

from lmc.config import graph_attrs
from lmc import viz
from lmc.core import ops, io

from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon


import pickle
import sys


def check_attrs(g: Graph) -> bool:
    """Check consistence of Graph attributes.

    Args:
        graph: The `igraph.Graph` object to be checked.

    Returns:
        Whether the attributes are the same as required.

    Raises:
        When graph attributes are not consistent with required.
    """
    current_attr = {
        "graph": g.attributes(),
        "vertex": g.vs.attributes(),
        "edge": g.es.attributes(),
    }

    target_attr = graph_attrs

    attributes_missing = []
    attributes_excessive = []
    is_consistent = True

    if (len(set(target_attr["graph"])) != len(target_attr["graph"]) or
            len(set(target_attr["vertex"])) != len(target_attr["vertex"]) or
            len(set(target_attr["edge"])) != len(target_attr["edge"])):
        is_consistent = False
        print("List of target edge attributes not unique.")
        return is_consistent

    # Check if all target attributes are in current graph
    for k in target_attr.keys():
        for attr in target_attr[k]:
            if attr in current_attr[k]:
                continue
            else:
                attributes_missing.append(attr)
                is_consistent = False

    # Check if no additional attributes are in current graph
    for k in current_attr.keys():
        for attr in current_attr[k]:
            if attr in target_attr[k]:
                continue
            else:
                attributes_excessive.append(attr)
                is_consistent = False

    if not is_consistent:
        raise ValueError(
            "Attributes of graph not match, see `check_attrs`\n"
            f"\tMissing attribute(s): {attributes_missing}\n"
            f"\tExcessive attribute(s): {attributes_excessive}\n"
        )

    return is_consistent


def append_vertices(
    g: Graph,
    vertices: Vertices,
    col_defaults: dict
) -> None:
    """Append vertices with column default values to a graph."""
    g.add_vertices(vertices.shape[0], attributes = {
        "x": vertices[:, 0], "y": vertices[:, 1],
        "z": vertices[:, 0] if vertices.shape[1] == 3
            else np.zeros(vertices.shape[0], dtype=np.float64)
    } | col_defaults)


def refine_DA_density(
    g: Graph,

    ghp_boundary_offset: float = 800,
    ghp_simplify_tolerance: float = 10,

    nr_new_DAs_max: int = 50,
    min_DA_DA_distance: float = 150.,
    newDA_prior_target_density: float = 4.0,
    nr_tries_max: int = 1000,

    save_refinement_steps: bool = False,
    save_init_final_distribution: bool = False,
    save_path: str = "",
    show_MCA_ACA_root: bool = False
) -> VoronoiExt:
    """Refine the SA network, as descripted in
    S1 Appendix. Refinement of surface artery network.

    Args:
        graph: Graph to be refined.
        ghp_boundary_offset: See `_gen_ghost_points()`.
        ghp_simplify_tolerance: See `_gen_ghost_points()`.

    Returns:
        xy_DA_roots_new_points: New DA roots.
        polygon_vs_xy: Vertices of polygons.
    """
    path = Path(f"{save_path}refinement/")
    path.mkdir(parents=True, exist_ok=True)

    DA_roots = ops.filter_vs(g, "is_DA_root")

    # Prepare two arrays for new DA roots
    DA_roots_new: Vertices = np.empty((0, 2))
    is_initial_DA: NDArray[np.bool_] = np.full(DA_roots.shape[0],
                                               fill_value=True)

    # Voronoi tessalation with DA root coordinates
    # Create ghost points as boundary
    ghost_points = ops.gen_ghost_points(
        DA_roots, ghp_boundary_offset, ghp_simplify_tolerance
    )
    vor_initial = ops.voronoi_tessalation(DA_roots, ghost_points, "DA root")

    # Repeat refinement not allowed
    if "refinement" in g.attributes():
        print("Graph has been refined already.")
        return vor_initial

    vor = vor_initial

    for i_step in range(nr_new_DAs_max):
        print(f"Sampling new DA root #{i_step}")

        # Area of each polygon, sorted in order of input points
        _, area_vor_pt_regions = ops.get_areas_voronoi_polygons(vor)

        is_valid_point, xy_new_point = ops.sample_new_DA_point(
            vor, vor_initial,
            min_DA_DA_distance,
            newDA_prior_target_density,
            nr_tries_max=nr_tries_max
        )

        if is_valid_point and i_step < nr_new_DAs_max:

            DA_roots = np.vstack((DA_roots, xy_new_point))
            is_initial_DA = np.append(is_initial_DA, False)
            DA_roots_new = np.vstack((DA_roots_new, xy_new_point))

            _, i_voronoi_newpt_corr_regionpt = \
                ops.find_voronoi_region_contains_pt(vor, xy_new_point)

            if (save_refinement_steps or
                    (save_init_final_distribution and i_step == 0)):

                viz.plot_DAs_refinement(
                    g, DA_roots_new, vor,
                    show_areas=True,
                    area_vor_input_point_based=area_vor_pt_regions,
                    xy_new_point=xy_new_point,
                    i_voronoi_newpt_corr_regionpt=\
                        i_voronoi_newpt_corr_regionpt,
                    show_MCA_ACA_root=show_MCA_ACA_root,
                    show_new_point=True,
                    show_vs_ids=False,
                    title=f"DAs refinement step {i_step}",
                    filepath=f"{save_path}refinement/step#{i_step}.png"
                )

                areas_valid = area_vor_pt_regions[
                    (area_vor_pt_regions > 0) & (vor.point_annot == "DA root")
                ]
                viz.plot_DA_distribution(
                    areas_valid,
                    filepath_density=\
                        f"{save_path}refinement/density_distr#{i_step}.png",
                    filepath_area=\
                        f"{save_path}refinement/area_distr#{i_step}.png"
                )

                # Add new DA root to the Voronoi tessalation
                vor = ops.voronoi_tessalation(DA_roots, ghost_points, "DA root")

        else:
            if save_refinement_steps or save_init_final_distribution:
                viz.plot_DAs_refinement(
                    g, DA_roots_new, vor,
                    show_areas=True,
                    show_MCA_ACA_root=show_MCA_ACA_root,
                    area_vor_input_point_based=area_vor_pt_regions,
                    show_new_point=False,
                    xy_new_point=xy_new_point,
                    i_voronoi_newpt_corr_regionpt=np.intp(-1),
                    show_vs_ids=False,
                    title=f"DAs refinement step {i_step}",
                    filepath=f"{save_path}refinement/step#{i_step}.png"
                )
                areas_valid = area_vor_pt_regions[
                    (area_vor_pt_regions > 0) & (vor.point_annot == "DA root")
                ]
                viz.plot_DA_distribution(
                    areas_valid,
                    filepath_density=\
                        f"{save_path}refinement/density_distr#{i_step}.png",
                    filepath_area=\
                        f"{save_path}refinement/area_distr#{i_step}.png"
                )

            print(f"No valid new point found at refinement step #{i_step}")
            break

    # TODO?: merge Graph and Voronoi {{{

    ### Add Voronoi tessalation to graph.
    #> 1
    append_vertices(g, DA_roots_new, {
        "is_DA_root": True,
        "is_DA_root_added_manually": True,
        "type": "DA root added manually",
    })
    # TODO: add test
    #  vor.points = original DA roots + DA roots added + ghost points

    # #> 2
    # append_vertices(g, vor.points[vor.point_annot == "Ghost point"],
    #                 {"type": "Ghost point"})
    # # now g.vs = other + original DA roots + DA roots added + ghost points
    # #            ^^^^^^^^^^^^^^^^^^^^^^^^^
    # #            see io.create()
    # #                                   1 -> ^^^^^^^^^^^^^^
    # #                                                    2 -> ^^^^^^^^^^^^
    # #            + vor vertices (below)
    # append_vertices(g, vor.vertices, {"type": "Voronoi vertex"})

    # n_other = np.sum(np.array(g.vs["is_DA_root"]) == False)
    # n_vor_pts = vor.points.shape[0]
    #
    # es_vor_pts = vor.ridge_points + n_other
    #
    # es_vor_vs = [p for p in vor.ridge_vertices if p[0] != -1 and p[1] != -1]
    # es_vor_vs = np.array(es_vor_vs) + (n_other + n_vor_pts)
    #
    # g.add_edges(es_vor_pts, attributes={"type": "Voronoi point ridge"}) # ?
    # g.add_edges(es_vor_vs, attributes={"type": "Voronoi vertex ridge"})

    # if "Voronoi vertex" in g.vs["type"]:
    #     print("Repeatly adding Voronoi tessalation is not supported.")

    # }}} TODO?: merge Graph and Voronoi

    ### Add refinement attribute
    g["refinement"] = f"Arguments:\n" \
        f"ghp_boundary_offset: {ghp_boundary_offset}\n" \
        f"ghp_simplify_tolerance: {ghp_simplify_tolerance}\n" \
        f"nr_new_DAs_max: {nr_new_DAs_max}\n" \
        f"min_DA_DA_distance: {min_DA_DA_distance}\n" \
        f"newDA_prior_target_density: {newDA_prior_target_density}\n" \
        f"nr_tries_max: {nr_tries_max}\n" \
        f"ghp_boundary_offset: {ghp_boundary_offset}\n"

    return vor


def add_AVs(
    g: Graph, vor: VoronoiExt,
    target_da_av_ratio: float = 3,
    min_dist_2_DA: float = 100,
    min_dist_2_AV: float = 120,
    nr_tries_max: int = 10000
) -> None:
    """Short description here, following details below

    Args:
        g: Graph object which the AVs will be added to.
        vor: The Voronoi tessalation, constructed from DA roots (original and
            manually added) in `g`, according to which the AVs are sampled.
        target_da_av_ratio: Defaults to 3.
    """
    DA_roots= ops.filter_vs(g, vtype=("DA root", "DA root added manually"))

    regions = [
        vor.regions[i] for i in vor.point_region[vor.point_annot == "DA root"]
        if -1 not in vor.regions[i]
    ]
    ridges = np.unique(
        np.sort(
            np.vstack([
                np.vstack((r, np.roll(r, -1))).transpose(1, 0) for r in regions
            ]),
            axis=1
        ),
        axis=0
    )
    ridge_coords = np.stack(
        [(vor.vertices[rd[0]], vor.vertices[rd[1]]) for rd in ridges],
        axis = 0
    )
    ridge_lengths = np.linalg.norm(
        ridge_coords[:, 0] - ridge_coords[:, 1],
        axis=1
    )

    nr_AVs_new = np.round(
        target_da_av_ratio * np.sum(vor.point_annot == "DA root"), decimals= 0
    )

    # loop over all new DAs, target ratio, abort criteria
    i_new_AV = 0
    is_valid_AV = False
    xy_AV_roots = np.empty((0, 2), dtype=np.float64)
    for i_new_AV in range(nr_AVs_new):

        is_valid_AV, new_AV = ops.sample_new_AV_point(
            ridge_coords,
            ridge_lengths,
            DA_roots,
            xy_AV_roots,
            min_dist_2_DA,
            min_dist_2_AV,
            nr_tries_max=nr_tries_max)

        if is_valid_AV:
            xy_AV_roots = np.vstack((xy_AV_roots, new_AV.reshape(-1, 2)))
        else:
            print("No valid new point found after", i_new_AV, '/', nr_AVs_new,
                  'added')
            break

    if is_valid_AV:
        print("Total of", i_new_AV + 1, '/', nr_AVs_new, 'added')

    append_vertices(g, xy_AV_roots, {"type": "AV root"})


def connect_new_DA_roots(
    graph: Graph,
    min_dist_subpts: float = 100,
    max_len_new_vessel: float = 1000.0,
    distort_max: float = 0.0
) -> bool:
    vid_new_DAs = [
        vid for vid in graph.vs.indices
        if graph.vs["type"][vid] == "DA root added manually"
    ]

    nr_of_new_DAs = len(vid_new_DAs)

    is_successful = False

    for i in range(nr_of_new_DAs):
        vid_new_DA = vid_new_DAs[i]
        is_successful = _connect_new_DA_root_to_graph(
            graph, vid_new_DA,
            min_dist_subpts=min_dist_subpts,
            diameter_new_vessel = -1.0,
            max_len_new_vessel=max_len_new_vessel,
            distort_max=distort_max
        )
        if not is_successful:
            print("Could not connect all DAs due to error")
            break

    if is_successful:
        print(nr_of_new_DAs, "DAs added successfully")

    return is_successful


def _connect_new_DA_root_to_graph(
    graph: Graph,
    vid_new_DA: int,
    min_dist_subpts: float = 100,
    max_len_new_vessel: float = 1000.0,
    diameter_new_vessel: float = -1.0,
    distort_max: float = 0.0
) -> bool:
    """Connect a new DA root to a sub-point on a edge of the graph.

        |<------>|<------>| distort_max
                 |<--->| actually shifted distance (choose longer side)
    A============E=====d=============B                   -
                /^     ^                                /
               / |     |                               /
              /  |     original closest sub-point     /
             /   |                                   /  max_len_new_vessel
            /    final shifted sub-point            /
           /                                       /
          C <- new DA                             /
                                                 /
                                                -
    Args:
        graph: The Graph object (**modified in place**).
        xy_new_DA: Coordinates of the new DA root.
        min_dist_subpts: Min distance between new sub-points on original edges.
        max_len_new_vessel: Max length of CE.
        diameter_new_vessel: Diameter of CE.
            Values < 0 to use AB's original diameter.
        distort_max: Max shifting distance to randomly move d to E.
            Values <= 0 to disable distorting.

    Returns:
        Whether the new DA root is successfully connected.
    """
    if distort_max > 1.:
        raise ValueError("distort_max should be < 1")

    xy = ops.get_vs(graph)

    xy_new_DA = xy[vid_new_DA]

    # edges {{{
    nr_es_orig = graph.ecount()
    adj_list = np.array(graph.get_edgelist(), dtype=np.int_)
    # current length of all vessels (can be > the L2 distance between a and b)
    edge_lengths = np.array(graph.es['length'])
    edge_types = np.array(graph.es['type'])
    edge_is_collateral = np.array(graph.es['is_collateral']) == True
    # nr of sub-points on each edge
    edge_nr_new_subpts \
         = np.floor(edge_lengths / min_dist_subpts).astype(np.int_) - 1
    edge_nr_new_subpts[edge_nr_new_subpts < 0] = 0
    # }}}

    # vertices {{{
    nr_vs_orig = graph.vcount()
    # }}}

    # Create 1d arrays with all x and y coordinates, and the corresponding edge
    # id of all sub-points {{{
    _x_all_subpts = np.hstack([ # shape(number of all sub-points)
        np.linspace(
            start=xy[adj_list[edge_id, 0]][0], # x_from
            stop=xy[adj_list[edge_id, 1]][0], # x_to
            num=nr_subpt + 2
        )[1:-1] # [1,-1)
        for nr_subpt, edge_id in zip(edge_nr_new_subpts, np.arange(nr_es_orig))
    ])
    _y_all_subpts = np.hstack([ # shape(number of all sub-points)
        np.linspace(
            start=xy[adj_list[edge_id, 0]][1], # y_from
            stop=xy[adj_list[edge_id, 1]][1], # y_to
            num=nr_subpt + 2
        )[1:-1] # [1,-1)
        for nr_subpt, edge_id in zip(edge_nr_new_subpts, np.arange(nr_es_orig))
    ])
    # array of xy coordinates of all subpts
    xy_all_subpts = np.array([_x_all_subpts, _y_all_subpts]).T
    eids_all_subpts = np.hstack([
        np.array([edge_id] * nr_subpt, dtype=np.int_)
        for nr_subpt, edge_id in zip(edge_nr_new_subpts, np.arange(nr_es_orig))
    ])
    # }}}

    # get the closest sub-point {{{
    # L2 distance of new DA to all subpoints
    dist_new_DA_to_all_subpts = np.linalg.norm(xy_all_subpts - xy_new_DA,
                                               axis=1)
    dist_new_DA_to_all_subpts[
        # exclude PA edges and collateral edges
        edge_types[eids_all_subpts] != 0 | edge_is_collateral[eids_all_subpts]
    ] = np.inf

    # return False if not at least one empty subpoint is found
    if np.size(dist_new_DA_to_all_subpts) < 1:
        print("Unable to connect DA: not enough connection points on edges")
        return False

    # index of closest subpoint
    _i_d = np.argmin(dist_new_DA_to_all_subpts)
    # edge id and distance to new DA of the closest sub-point
    xy_d = xy_all_subpts[_i_d, :]
    eid_ab = eids_all_subpts[_i_d]
    # Vertex ids and coordinates of edge ends
    edge_vids_ab = adj_list[eid_ab, :]
    # coordinates of edge ends (a and b)
    xy_a = xy[edge_vids_ab[0]]
    xy_b = xy[edge_vids_ab[1]]
    # }}}

    # The new vertex on old edge is shifted, to make the network less
    # structured {{{
    xy_e = xy_d
    if distort_max > 0:
        len_abs_ad = np.linalg.norm(xy_d - xy_a)
        len_abs_db = np.linalg.norm(xy_b - xy_d)
        # direction vector specifying distorting range on edge
        vec_distort_max = (xy_d - xy_a) / len_abs_ad

        delta = np.random.rand() * distort_max
        if len_abs_ad > len_abs_db:
            xy_e = xy_d - vec_distort_max * delta * len_abs_ad
        else:
            xy_e = xy_d + vec_distort_max * delta * len_abs_db
    # }}}

    # set diameters and lengths of new vessels {{{
    length_ce = np.linalg.norm(xy_e - xy_new_DA)
    if length_ce > max_len_new_vessel:
        print("Unable to connect DA: distance (" + str(xy_e) \
            + " exceeds max distance ("+str(max_len_new_vessel)+")")
        return False
    length_ab = graph.es['length'][eid_ab]
    length_ae = length_ab * \
        np.linalg.norm(xy_e - xy_a) / np.linalg.norm(xy_b - xy_a)
    length_eb = length_ab * \
        np.linalg.norm(xy_b - xy_e) / np.linalg.norm(xy_b - xy_a)

    diameter_ab_orig = graph.es['diameter'][eid_ab]
    diameter_ae = diameter_eb = diameter_ab_orig
    diameter_ce = diameter_ab_orig if diameter_new_vessel < 0 \
        else diameter_new_vessel
    # }}}

    # add new vertices {{{
    # vertex E
    graph.add_vertex(x=xy_e[0], y=xy_e[1], type="Sub point",
                     is_DA_root=False, is_DA_root_added_manually=False)
    # }}}

    # add new edges {{{
    vid_e = nr_vs_orig
    vid_c = vid_new_DA
    is_added_manually_ab = graph.es['is_added_manually'][eid_ab]

    # edges AE and EB
    graph.delete_edges(eid_ab)
    graph.add_edge(edge_vids_ab[0], vid_e,
                   is_added_manually=is_added_manually_ab,
                   diameter=diameter_ae, length=length_ae, type=0)
    graph.add_edge(edge_vids_ab[1], vid_e,
                   is_added_manually=is_added_manually_ab,
                    diameter=diameter_eb, length=length_eb, type=0)
    # CE
    graph.add_edge(vid_c, vid_e, is_added_manually=True, diameter=diameter_ce,
                   length=length_ce, type=0)
    # }}}

    return True


def add_da_trees_to_sa_graph(graph_main: Graph, da_candidates) -> bool:
    nr_of_open_DA_starting_pts = 1e6

    graph_main.vs['is_free_root'] = graph_main.vs['is_DA_root']

    while nr_of_open_DA_starting_pts > 0:

        i_DA_tree = np.random.choice(da_candidates)
        rotation = np.random.rand() * 2 * np.pi

        g_da = io.load_penetrating_tree(
            ptr_type=2, id=i_DA_tree, scale=2 / 3, rotate=rotation,
            min_diam=4.5
        )

        print(f'DA tree: {i_DA_tree}, '
              f'Max. degree current DA: {np.max(g_da.degree())}')

        ####### HRER ########

        graph_main, nr_of_open_DA_starting_pts \
                               = _merge_sa_with_tree(graph_main, g_da, 2)

    del graph_main.vs['is_free_root']

    return graph_main

def _merge_sa_with_tree(
    graph_main: Graph, graph_tree: Graph, type_tree: Literal[2, 3]
) -> bool:
    """Merge SA network with penetrating tree."""

    # Read PA data {{{
    ## adjacency (contains vertex ids) != edge (contains vertex coordinates)
    adj_pa = np.array(graph_main.get_edgelist(), dtype=np.int_)
    vs_pa = ops.get_vs(graph_main, z=True)
    n_vs_pa = graph_main.vcount()

    ## choose free root {{{
    vid_pa_free_roots = np.where(graph_main.vs['is_free_root'])[0]
    nr_pa_free_roots = np.size(vid_pa_free_roots) # current DA root to connect

    print("Nr of remaining tree starting points:", nr_pa_free_roots)

    if nr_pa_free_roots < 1:
        return True

    vid_pa_free_root = vid_pa_free_roots[0]
    pa_free_root = vs_pa[vid_pa_free_root]
    ## }}}
    # }}}

    print(f"Connecting tree to vertex #{vid_pa_free_root}: {pa_free_root}")

    # Read tree data and vertex offset to PA graph {{{
    adj_tree = np.array(graph_tree.get_edgelist(), dtype=np.int_)
    n_adj_tree = graph_tree.ecount()
    n_vs_tree = graph_tree.vcount()
    vs_tree_moved = ops.get_vs(graph_tree, z=True) + pa_free_root # translate

    ## get attachment point (root point) {{{
    if type_tree == 2:
        vid_tree_root = np.where(graph_tree.vs['is_connected2PA'])[0] \
            + n_vs_pa
    elif type_tree == 3:
        vid_tree_root = np.where(graph_tree.vs['is_connected2PV'])[0] \
            + n_vs_pa
    if np.size(vid_tree_root) > 1:
        raise ValueError("More than 1 root point found in penetrating vessel "
                         f"tree {graph_tree["name"]}.")
    vid_tree_root = vid_tree_root[0] # reduce dimension
    ## }}}
    # }}}

    graph_main.add_vertices(
        vs_tree_moved.shape[0],
        attributes={
            "is_connected2PA": graph_tree.vs["is_connected2PA"],
            "is_connected2PV": graph_tree.vs["is_connected2PV"],
            "is_connected2caps": graph_tree.vs["is_connected2caps"],
            "x": vs_tree_moved[:, 0],
            "y": vs_tree_moved[:, 1],
            "z": vs_tree_moved[:, 2],
            "is_free_root": False,
            "type": "DA"
        }
    )
    # it is connected now, set to False
    graph_main.vs[2]["is_free_root"] = False

    graph_main.add_edges(
        adj_tree + n_vs_pa,
        attributes={
        "length": graph_tree.es["length"],
        "diameter": graph_tree.es["diameter"],
        "type": graph_tree.es["type"],
        "is_added_manually": True
    })

    # connect vessel tree to PA network {{{
    pt_connected_to_tree_root = graph_main.neighbors(vid_tree_root)
    if np.size(pt_connected_to_tree_root) > 1:
        raise ValueError("Found more than 1 neighbors (following points) of "
                         "tree root before connecting.")

    # find eid of edge adjacent to head node, first segment of vessel tree
    eid_1st_seg = graph_main.get_eid(vid_tree_root,
                                    pt_connected_to_tree_root[0])
    graph_main.add_edge(
        vid_pa_free_root, pt_connected_to_tree_root[0],
        length=graph_main.es['length'][eid_1st_seg],
        diameter=graph_main.es['diameter'][eid_1st_seg],
        type=graph_main.es['type'][eid_1st_seg],
        is_added_manually=True
    )  # add edge from PA to tree-neighbour of head node

    graph_main.delete_edges(eid_1st_seg)  # delete remaining edge
    graph_main.delete_vertices(vid_tree_root)  # delete remaining node
    # }}}

    return True


