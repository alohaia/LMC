"""Higher-level network manipulation toolkit.

*Higher-level* means these functions will be used directly in user-created
scripts. Mainly contains functions manipulating network objects (igraph.Graph).
"""

from pathlib import Path
from typing import Literal

import numpy as np
from igraph import Graph
from numpy.typing import NDArray
from scipy.spatial import KDTree
from shapely.lib import length

from lmc import config, viz
from lmc.config import graph_attrs
from lmc.core import io, ops
from lmc.types import Vertices, VoronoiExt


def check_attrs(g: Graph, preset: str) -> bool:
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

    target_attr = graph_attrs[preset]

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

    DA_roots = ops.filter_vs(g, "is_DA_root", z=False)

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
    DA_roots= ops.filter_vs(g, vtype=("DA root", "DA root added manually"),
                            z=False)

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

    append_vertices(g, xy_AV_roots, {
        "is_AV_root": True,
        "type": "AV root"
    })


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

    xy = ops.get_vs(graph, z=False)

    xy_new_DA = xy[vid_new_DA]

    # edges {{{
    nr_es_orig = graph.ecount()
    adj_list = np.array(graph.get_edgelist(), dtype=np.int_)
    # current length of all vessels (can be > the L2 distance between a and b)
    edge_lengths = np.array(graph.es['length'])
    edge_types = np.array(graph.es['type'])
    edge_is_collateral = np.array(graph.es['is_collateral']) == True
    edge_added_manually = np.array(graph.es["is_added_manually"])
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
    # exclude non-PA edges and collateral edges
    try:
        dist_new_DA_to_all_subpts[
            (edge_types[eids_all_subpts] != "PA")
            | edge_is_collateral[eids_all_subpts]
            | edge_added_manually[eids_all_subpts]
        ] = np.inf
    except(TypeError):
        breakpoint()

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
    edge_ab = adj_list[eid_ab, :]
    # coordinates of edge ends (a and b)
    xy_a = xy[edge_ab[0]]
    xy_b = xy[edge_ab[1]]
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

    # add new vertices {{{
    # vertex E
    graph.add_vertex(x=xy_e[0], y=xy_e[1], z=0, type="PA subpoint",
                     is_DA_root=False, is_DA_root_added_manually=False)
    # }}}

    # add new edges {{{
    vid_e = nr_vs_orig
    vid_c = vid_new_DA

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

    # edges AE and EB
    graph.add_edges(
        [
            [edge_ab[0], vid_e], # AE
            [edge_ab[1], vid_e], # BE
            [vid_c, vid_e]       # CE
        ],
        attributes={
            'type': 'PA',
            'length': [length_ae, length_eb, length_ce],
            'is_added_manually': False,
        } | {
            dattr: [
                graph.es[eid_ab][dattr],
                graph.es[eid_ab][dattr],
                graph.es[eid_ab][dattr]
                    if diameter_new_vessel < 0 else diameter_new_vessel
            ]
            for dattr in config.diameter_attrs
        }
    )
    graph.delete_edges(eid_ab)
    # }}}
    # }}}

    return True


def add_pt_trees(
    graph_main: Graph,
    tree_type: list[Literal["DA", "AV"]]
) -> bool:
    """Add penetrating trees to graph."""
    for ttype in tree_type:

        graph_main.vs['is_free_root'] = \
            np.equal(graph_main.vs[f"is_{ttype}_root"], True)

        while np.sum(graph_main.vs['is_free_root']) > 0:
            i_free_root = np.random.choice(
                config.pttree_candidates[ttype]
            )
            rotation = np.random.rand() * 2 * np.pi

            g_pttree = io.load_penetrating_tree(
                tree_type=ttype,
                id=i_free_root,
                scale=2 / 3,
                rotate=rotation,
                min_diam=4.5
            )

            print(f"{ttype} tree: {i_free_root}, "
                f"Max. degree: {np.max(g_pttree.degree())}")

            _merge_sa_with_tree(graph_main, g_pttree, ttype)

        del graph_main.vs['is_free_root']

    return True


def _merge_sa_with_tree(
    graph_main: Graph, graph_tree: Graph, tree_type: Literal["DA", "AV"]
) -> bool:
    """Merge SA network with penetrating tree."""

    # Read PA data {{{
    ## adjacency (contains vertex ids) != edge (contains vertex coordinates)
    vs_pa = ops.get_vs(graph_main, z=True)
    n_vs_pa = graph_main.vcount()

    ## choose free root {{{
    vid_pa_free_roots = np.where(graph_main.vs['is_free_root'])[0]
    n_pa_free_roots = np.size(vid_pa_free_roots) # current DA root to connect

    print("Nr of remaining tree starting points:", n_pa_free_roots)

    if n_pa_free_roots < 1:
        return True

    vid_pa_free_root = vid_pa_free_roots[0]
    pa_free_root = vs_pa[vid_pa_free_root]
    ## }}}
    # }}}

    print(f"Connecting tree to vertex #{vid_pa_free_root}: {pa_free_root}")

    # Read tree data and vertex offset to PA graph {{{
    adj_tree = np.array(graph_tree.get_edgelist(), dtype=np.int_)
    vs_tree_moved = ops.get_vs(graph_tree, z=True) + pa_free_root # translate

    ## get attachment point (root point) {{{
    if tree_type == "DA":
        vid_tree_roots = np.where(graph_tree.vs['is_connected2PA'])[0] \
            + n_vs_pa
    elif tree_type == "AV":
        vid_tree_roots = np.where(graph_tree.vs['is_connected2PV'])[0] \
            + n_vs_pa

    if np.size(vid_tree_roots) > 1:
        raise ValueError('More than 1 root point found in penetrating vessel '
                         f'tree {graph_tree["name"]}.')

    vid_tree_root = vid_tree_roots[0] # reduce dimension to single value
    ## }}}
    # }}}

    graph_main.add_vertices(
        vs_tree_moved.shape[0],
        attributes={
            'is_connected2PA': graph_tree.vs['is_connected2PA'],
            'is_connected2PV': graph_tree.vs['is_connected2PV'],
            'is_connected2Cap': graph_tree.vs['is_connected2Cap'],
            'x': vs_tree_moved[:, 0],
            'y': vs_tree_moved[:, 1],
            'z': vs_tree_moved[:, 2],
            'is_free_root': False,
            'type': f'{tree_type} tree point'
        }
    )
    # it is connected now, set to False
    graph_main.vs[vid_pa_free_root]["is_free_root"] = False

    graph_main.add_edges(
        adj_tree + n_vs_pa,
        attributes={
        "length": graph_tree.es["length"],
        "type": graph_tree.es["type"],
    } | {
        dattr: graph_tree.es["diameter"] for dattr in config.diameter_attrs
    })

    # connect vessel tree to PA network {{{
    pt_connected_to_tree_root = graph_main.neighbors(vid_tree_root)
    if np.size(pt_connected_to_tree_root) > 1:
        raise ValueError("Found more than 1 neighbors (following points) of "
                         "tree root before connecting.")

    # find eid of edge adjacent to head node, first segment of vessel tree
    eid_1st_seg = graph_main.get_eid(vid_tree_root,
                                     pt_connected_to_tree_root[0])
    graph_main.add_edges(
        [[vid_pa_free_root, pt_connected_to_tree_root[0]]],
        attributes={
            'length': graph_main.es['length'][eid_1st_seg],
            'type': graph_main.es['type'][eid_1st_seg]
        } | {
            dattr: graph_main.es['diameter'][eid_1st_seg]
            for dattr in config.diameter_attrs
        }
    )  # add edge from PA to tree-neighbour of head node

    graph_main.delete_edges(eid_1st_seg)  # delete remaining edge
    graph_main.delete_vertices(vid_tree_root)  # delete remaining node
    # }}}

    return True

def add_capillary_bed(
    g_main: Graph,
    z_min_caps=25.0,
    frame_bounding_box=50.0,
    distance_between_cap_vs=45.0,
    l_vessel=62.0,
    d_vessel=4.5,
    perturb_vs_frac = 0.0
) -> None:
    # Find bounding box coordinates of capillary bed
    coords_connection_pts = ops.filter_vs(g_main, "is_connected2Cap", z=True)

    x_min = np.min(coords_connection_pts[:, 0]) - frame_bounding_box
    x_max = np.max(coords_connection_pts[:, 0]) + frame_bounding_box

    y_min = np.min(coords_connection_pts[:, 1]) - frame_bounding_box
    y_max = np.max(coords_connection_pts[:, 1]) + frame_bounding_box

    z_min = z_min_caps
    z_max = np.max(coords_connection_pts[:, 2]) + frame_bounding_box

    print(f"Bounding box: ({x_min}~{x_max}, {y_min}~{y_max}, {z_min}~{z_max})")

    g_capbed = ops.create_stacked_hex_network(
        x_min, x_max, y_min, y_max, z_min, z_max,
        distance_between_cap_vs, l_vessel, d_vessel, perturb_vs_frac
    )
    g_capbed.es["type"] = "Cap"
    g_capbed.vs["type"] = "Cap point"

    _merge_with_capbed(g_main, g_capbed)


def _merge_with_capbed(g_main: Graph, g_capbed:Graph) -> None:
    from datetime import datetime

    n_vs_main_orig = g_main.vcount()
    g_main.add_vertices(
        g_capbed.vcount(),
        attributes={a: g_capbed.vs[a] for a in g_capbed.vs.attributes()}
    )
    g_main.add_edges(
        np.array(g_capbed.get_edgelist()) + n_vs_main_orig,
        attributes={a: g_capbed.es[a] for a in g_capbed.es.attributes()}
    )

    # leaf points
    leafs = ops.filter_vs(g_main, attr="is_connected2Cap")
    vid_leafs = np.where(np.equal(g_main.vs["is_connected2Cap"], True))[0]

    # mid-points of caps edges
    caps_midpts = np.sum(ops.filter_es(g_main, etype="Cap"), axis=1) / 2
    eid_caps = np.where(np.equal(g_main.es["type"], "Cap"))[0]

    # # method 1: use KDTree to caculate min distances
    # kdtree_midpts = KDTree(caps_midpts)
    # _, indices = kdtree_midpts.query(leafs, k=1)

    # # # method 2: find global optimal solution with no duplicate points
    # # dist_matrix = cdist(leafs, caps_midpts)  # shape: (n_leafs, n_midpts)
    # # _, indices = linear_sum_assignment(dist_matrix) # closest vertex indices

    # # (method 1 and 2)connect leafs to end-points of closest caps
    # adj_list_merged = np.array(g_main.get_edgelist(), dtype=np.int_)
    # eid_closest_caps = eid_caps[indices]
    #
    # es_leaf2endpts = np.vstack((
    #     np.vstack((vid_leafs, adj_list_merged[eid_closest_caps][:, 0])).T,
    #     np.vstack((vid_leafs, adj_list_merged[eid_closest_caps][:, 1])).T
    # ))
    #
    # g_main.add_edges(
    #     es=es_leaf2endpts,
    #     attributes={
    #         "length": np.array(g_main.es["length"])[eid_closest_caps],
    #         "type": "Cap",
    #     } | {
    #         # "diameter": np.array(g_main.es["diameter"])[eid_closest_caps],
    #         # config.diameter_attrs
    #     }
    # )
    # g_main.delete_edges(np.unique(eid_closest_caps))

    # method 3: update network after connecting each leaf point
    _leaf_number_len = len(str(leafs.shape[0]))
    for i_leaf in range(leafs.shape[0]):

        vid_leaf = vid_leafs[i_leaf]

        caps_midpts = np.sum(ops.filter_es(g_main, etype="Cap"), axis=1) / 2
        eid_caps = g_main.es(type_eq="Cap").indices
        dists = np.linalg.norm(caps_midpts - leafs[i_leaf], axis=1)
        e_closest_caps = g_main.es[eid_caps[np.argmin(dists)]]

        g_main.add_edges(
            es=np.array([
                [vid_leaf, e_closest_caps.source],
                [vid_leaf, e_closest_caps.target],
            ]),
            attributes={
                "length": e_closest_caps["length"],
                "type": "Cap",
            } | {
                dattr: e_closest_caps["diameter"] for dattr in config.diameter_attrs
            }
        )

        if i_leaf % 100 == 0:
            timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            print(
                f'[{timestamp}] Connecting leafs: '
                f'{i_leaf:>{_leaf_number_len}d} / {leafs.shape[0]} completed: '
                f'leaf({vid_leaf}) to '
                f'edge({eid_caps[np.argmin(dists)]}: '
                f'{e_closest_caps.source}->{e_closest_caps.target})...'
            )


def remove_redundant_vessels(graph: Graph) -> None:
    # TODO:
    # 1. Closure
    pass


def remove_regions_outside_DAs(
    graph: Graph,
    max_distance_to_da: float
) -> None:
    vid_all = np.arange(graph.vcount())
    vs_all = ops.get_vs(graph, z=True)

    # vkind is type for vertices
    vkind = np.array(graph.vs["type"]).astype("<U2")
    # exclude AV/DA roots
    vkind[np.isin(graph.vs["type"], ("AV root", "DA root",
                                     "DA root added manually"))] = ""

    # (Pdb) pd.DataFrame(np.unique(graph.vs["type"], return_counts=True))
    #              0              1          2        3                       4
    # 0      AV root  AV tree point  Cap point  DA root  DA root added manually
    # 1          180          21654      43776       18                      42
    #               5             6            7
    # 0 DA tree point      PA point  PA subpoint
    # 1          5607            36           42
    # (Pdb) pd.DataFrame(np.unique(vkind, return_counts=True))
    #        0      1     2   3
    # 0            AV    DA  PA
    # 1  43776  21834  5667  78

    # get distances of each AV vertex to its closest DA vertices
    tree_DAs = KDTree(vs_all[vkind == "DA"])
    dist_AVs2closestDA, _ = tree_DAs.query(vs_all[vkind == "AV"])
    # delete all AVs which are beyond certain distance to DAs
    vid_AVs = vid_all[vkind == "AV"]
    vid_far_AVs = vid_AVs[dist_AVs2closestDA > max_distance_to_da]
    graph.delete_vertices(vid_far_AVs)

    # tree_ = KDTree(vs_all[np.isin(vkind, ("DA", "AV"))])
    # dist_AVs2closestDA, _ = tree_DAs.query(vs_all[vkind == "Ca"])
    # # delete all AVs which are beyond certain distance to DAs
    # vid_AVs = vid_all[vkind == "AV"]
    # vid_far_AVs = vid_AVs[dist_AVs2closestDA > max_distance_to_da]
    # graph.delete_vertices(vid_far_AVs)

    # Delete resulting capillary and AV (exclude roots) dead Ends
    graph.vs['vkind'] = vkind
    graph.vs['degree'] = graph.degree()
    while len(graph.vs(degree_eq=1, vkind_in=("AV", "Ca"))) != 0:
        graph.delete_vertices(graph.vs(degree_eq=1,
                                       vkind_in=("AV", "Ca")).indices)
        graph.vs['degree'] = graph.degree() # update degree attribute

    # delete degree 0 vertices (vertices that are now completely disconnected)
    graph.delete_vertices(graph.vs(degree_eq=0, vkind_eq="AV").indices)

    del graph.vs['degree'], graph.vs['vkind']

    print('Network cropped')


def check_microbloom_attrs(graph):
    vattrs = graph.vs.attributes()
    assert 'coords' in vattrs
    assert 'boundaryType' in vattrs
    assert 'boundaryValue' in vattrs

    eattrs = graph.es.attributes()
    for dattr in config.diameter_attrs:
        assert dattr in eattrs
    assert 'length' in eattrs

    components = graph.components()
    if len(components) > 1:
        print(f'{len(components)} components detected.')
    for comp in components:
        vs = list(comp)
        has_pressure = np.any(
            np.isin(vs, graph.vs(boundaryType_eq=1).indices)
        )
        if not has_pressure:
            print("Component without pressure BC:", vs)



