import numpy as np
import igraph
from scipy.spatial import KDTree
import sys

from import_graph import import_da_graph_from_pkl, import_av_graph_from_pkl
from generate_capillary_bed import create_stacked_hex_network

def connect_all_new_DAs_starting_pt_to_graph(graph, xy_new_DAs,
                                             target_new_min_splitedgelength=100,
                                             max_distance_new_DA=1000., distort=False, distort_max=0.):

    nr_of_new_DAs = np.size(xy_new_DAs, 0)

    is_valid = False

    for i in xrange(nr_of_new_DAs):

        xy_new_DA = xy_new_DAs[i, :]

        graph_new, is_valid = connect_new_DA_starting_pt_to_graph(graph, xy_new_DA,
                                                                  target_new_min_splitedgelength=target_new_min_splitedgelength,
                                                                  diameter_new_vessel = -1.,
                                                                  max_distance_new_DA=max_distance_new_DA,
                                                                  distort=distort, distort_max=distort_max)
        if not is_valid:
            print "Could not connect all DAs due to error"
            break

    if is_valid:
        print nr_of_new_DAs, "DAs added successfully"

    return graph, is_valid


def connect_new_DA_starting_pt_to_graph(graph, xy_new_DA, target_new_min_splitedgelength = 100, max_distance_new_DA = 1000., diameter_new_vessel = -1., distort = False, distort_max = 0.):

    x = np.array(graph.vs["coords"])[:, 0]
    y = np.array(graph.vs["coords"])[:, 1]
    xy = np.array([x, y]).transpose() # array of xy coordinates

    nr_of_es_old = graph.ecount()
    nr_of_vs_old = graph.vcount()

    adjacency_list = np.array(graph.get_edgelist(), dtype=np.int)

    # current length of all vessels (can be > the L2 distance between a and b)
    length_ab = np.array(graph.es['length'])
    type_ab = np.array(graph.es['type'])
    is_collateral = np.array(graph.es['is_collateral']) > 0

    # compute nr of candidates for connecting the new DA on each edge (nr of sub-points)
    nr_new_subpts_ab = np.floor(length_ab / target_new_min_splitedgelength).astype(np.int) - 1
    nr_new_subpts_ab[nr_new_subpts_ab < 0] = 0

    # Create 1d arrays with all x and y coordinates, and the corresponding edge id of all sub-points
    x_all_subpts = np.hstack(
        [np.linspace(x[adjacency_list[edge_id, 0]], x[adjacency_list[edge_id, 1]], current_nr_subpt + 2)[1:-1] for
         current_nr_subpt, edge_id in zip(nr_new_subpts_ab, np.arange(nr_of_es_old))])

    y_all_subpts = np.hstack(
        [np.linspace(y[adjacency_list[edge_id, 0]], y[adjacency_list[edge_id, 1]], current_nr_subpt + 2)[1:-1] for
         current_nr_subpt, edge_id in zip(nr_new_subpts_ab, np.arange(nr_of_es_old))])

    xy_all_subpts = np.array([x_all_subpts, y_all_subpts]).transpose() # array of xy coordinates of all subpts
    eids_all_subpts = np.hstack(
        [np.ones(current_nr_subpt, dtype=np.int)*edge_id for current_nr_subpt, edge_id in zip(nr_new_subpts_ab, np.arange(nr_of_es_old))])

    # L2 distance of new DA to all subpoints
    distance_new_DA_to_all_subpts = np.linalg.norm(xy_all_subpts - xy_new_DA, axis=1)

    distance_new_DA_to_all_subpts[type_ab[eids_all_subpts] != 0] = 10.*max_distance_new_DA
    distance_new_DA_to_all_subpts[is_collateral[eids_all_subpts]] = 10.*max_distance_new_DA

    # return False if not at least one empty subpoint is found
    if np.size(distance_new_DA_to_all_subpts) < 1:
        print "Unable to connect DA: not enough connection points on edges"
        return graph, False

    # index of closest subpoint
    index_closest_subpt = np.argmin(distance_new_DA_to_all_subpts)

    # identify edge id and distance of closest sub point
    eid_old_ab = eids_all_subpts[index_closest_subpt]
    r_d_closest = xy_all_subpts[index_closest_subpt, :]

    # Vertex ids and coordinates of edge ends
    vids_old_ab = adjacency_list[eid_old_ab, :]

    r_a_final = xy[vids_old_ab[0]]
    r_b_final = xy[vids_old_ab[1]]

    # The new vertex on old edge is shifted, to make the network less structured
    if distort:
        r_ad_final_abs = np.linalg.norm(r_d_closest - r_a_final)
        r_db_final_abs = np.linalg.norm(r_b_final - r_d_closest)
        e_ad_final = (r_d_closest - r_a_final) / r_ad_final_abs # direction vector on edge

        if distort_max > 1.:
            print "Warning: distort_max should be < 1"
            return graph, False

        delta = np.random.rand() * distort_max
        if r_ad_final_abs > r_db_final_abs:
            r_e_closest = r_d_closest - e_ad_final * delta * r_ad_final_abs
        else:
            r_e_closest = r_d_closest + e_ad_final * delta * r_db_final_abs
    else:
        r_e_closest = r_d_closest

    # set diameters and lengths of new vessels
    diameter_old_ab = graph.es['diameter'][eid_old_ab]
    length_old_ab = graph.es['length'][eid_old_ab]
    added_manually_old_ab = graph.es['added_manually'][eid_old_ab]

    if diameter_new_vessel < 0:
        diameter_new_ce = diameter_old_ab
    else:
        diameter_new_ce = diameter_new_vessel

    diameter_new_ae = diameter_new_eb = diameter_old_ab

    length_new_ce = np.linalg.norm(r_e_closest - xy_new_DA)

    length_new_ae = length_old_ab * np.linalg.norm(r_e_closest - r_a_final) / np.linalg.norm(r_b_final - r_a_final)
    length_new_eb = length_old_ab * np.linalg.norm(r_b_final - r_e_closest) / np.linalg.norm(r_b_final - r_a_final)

    if length_new_ce > max_distance_new_DA:
        print "Unable to connect DA: distance ("+str(r_e_closest)+" exceeds max distance ("+str(max_distance_new_DA)+")"
        return graph, False

    # add the new vertices

    graph.add_vertex(ACA_in=0, COW_in=0, MCA_in=0, coords=[xy_new_DA[0], xy_new_DA[1], 0.], is_connected_2caps=0,
                     is_AV_root=0, is_DA_startingPt=1, is_DA_startingPt_added_manually=1)  # new vertex "C" (new DA)
    graph.add_vertex(ACA_in=0, COW_in=0, MCA_in=0, coords=[r_e_closest[0], r_e_closest[1], 0.], is_connected_2caps=0,
                     is_AV_root=0, is_DA_startingPt=0, is_DA_startingPt_added_manually=0)  # new vertex "E"

    v_id_new_C = nr_of_vs_old
    v_id_new_E = nr_of_vs_old + 1

    # connect new edges to new vertices
    # Edge CE
    graph.add_edge(v_id_new_C, v_id_new_E, added_manually=1, diam_post_exp=-1, diam_pre_exp=-1,
                   diameter=diameter_new_ce, index_exp=-1, is_collateral=0, is_stroke=0, length=length_new_ce, type=0,
                   vRBC_post_exp=None, vRBC_pre_exp=None, vRBC_pre_larger10=0)

    # edges AE and EB; assign measurements only to one edge (choose randomly based on lengths)
    diam_post_exp_old_ab = graph.es['diam_post_exp'][eid_old_ab]
    diam_pre_exp_old_ab = graph.es['diam_pre_exp'][eid_old_ab]
    index_exp_old_ab = graph.es['index_exp'][eid_old_ab]
    vRBC_pre_larger10_old_ab = graph.es['vRBC_pre_larger10'][eid_old_ab]

    if np.random.rand() < length_new_ae / (length_new_ae+length_new_eb):

        # assign measurements to edge AE
        # direction of measurement remains {since eid(A)<eid(B) and eid(A)<eid(E)}
        vRBC_post_exp_ae = graph.es['vRBC_post_exp'][eid_old_ab]
        vRBC_pre_exp_ae = graph.es['vRBC_pre_exp'][eid_old_ab]

        # assign measurements to edge AE
        # direction of measurement remains {eid(A)<eid(B) and eid(A)<eid(E)
        graph.add_edge(vids_old_ab[0], v_id_new_E, added_manually=added_manually_old_ab,
                       diam_post_exp=diam_post_exp_old_ab,
                       diam_pre_exp=diam_pre_exp_old_ab, diameter=diameter_new_ae, index_exp=index_exp_old_ab,
                       is_collateral=0, is_stroke=0, length=length_new_ae, type=0, vRBC_post_exp=vRBC_post_exp_ae,
                       vRBC_pre_exp=vRBC_pre_exp_ae, vRBC_pre_larger10=vRBC_pre_larger10_old_ab)

        graph.add_edge(vids_old_ab[1], v_id_new_E , added_manually=added_manually_old_ab,
                       diam_post_exp=-1, diam_pre_exp=-1, diameter=diameter_new_eb, index_exp=-1,
                       is_collateral=0, is_stroke=0, length=length_new_eb, type=0,
                       vRBC_post_exp=None, vRBC_pre_exp=None, vRBC_pre_larger10=0)

    else:
        # assign measurements to edge BE
        # direction of measurement changes {since eid(B)<eid(E) but eid(B)>eid(A)}
        vRBC_post_exp_be = graph.es['vRBC_post_exp'][eid_old_ab]
        vRBC_pre_exp_be = graph.es['vRBC_pre_exp'][eid_old_ab]

        if vRBC_post_exp_be is not None:
            vRBC_post_exp_be *= (-1.)

        if vRBC_pre_exp_be is not None:
            vRBC_pre_exp_be *= (-1.)

        # assign measurements to edge EB
        graph.add_edge(vids_old_ab[0], v_id_new_E, added_manually=added_manually_old_ab,
                       diam_post_exp=-1, diam_pre_exp=-1, diameter=diameter_new_ae, index_exp=-1,
                       is_collateral=0, is_stroke=0, length=length_new_ae, type=0, vRBC_post_exp=None,
                       vRBC_pre_exp=None, vRBC_pre_larger10=0)

        graph.add_edge(vids_old_ab[1], v_id_new_E, added_manually=added_manually_old_ab,
                       diam_post_exp=diam_post_exp_old_ab,
                       diam_pre_exp=diam_pre_exp_old_ab, diameter=diameter_new_eb, index_exp=index_exp_old_ab,
                       is_collateral=0, is_stroke=0, length=length_new_eb, type=0, vRBC_post_exp=vRBC_post_exp_be,
                       vRBC_pre_exp=vRBC_pre_exp_be, vRBC_pre_larger10=vRBC_pre_larger10_old_ab)

    # remove old edge
    graph.delete_edges(eid_old_ab)

    return graph, True

def add_AV_starting_pts_to_graph(graph, xy_AVs):

    nr_of_AVs = np.size(xy_AVs, 0)

    for i in xrange(nr_of_AVs):
        graph.add_vertex(ACA_in=0, COW_in=0, MCA_in=0, coords=[xy_AVs[i,0], xy_AVs[i,1], 0.], is_connected_2caps=0,
                         is_AV_root=1, is_DA_startingPt=0, is_DA_startingPt_added_manually=0)  # new AV vertex

    return graph


def add_da_trees_to_sa_graph(graph_parent, da_candidates, path_trees):

    nr_of_open_DA_starting_pts = 1e6

    graph_parent.vs['is_tree_startingPt_free'] = graph_parent.vs['is_DA_startingPt']

    while nr_of_open_DA_starting_pts > 0:

        current_DA_id = np.random.choice(da_candidates)
        current_rotation = np.random.rand() * 2 * np.pi

        path_artery_es = path_trees + str(current_DA_id) + '_edgesDict.pkl'
        path_artery_vs = path_trees + str(current_DA_id) + '_verticesDict.pkl'

        g_da = import_da_graph_from_pkl(path_artery_es, path_artery_vs, rotation=current_rotation,
                                        l_scaling=2. / 3., min_diam=4.5, adj_attr_name='tuple', graph_name='B6 DA')

        print 'DA tree:', current_DA_id, 'Max. degree current DA:', np.max(g_da.degree())
        graph_parent, nr_of_open_DA_starting_pts = merge_sa_with_trees(graph_parent, g_da, 2)

    del graph_parent.vs['is_tree_startingPt_free']

    return graph_parent


def merge_sa_with_trees(g_main, g_daughter, type_daughter):

    # Read PA data
    adj_list_pa = np.array(g_main.get_edgelist(), dtype=np.int)
    nr_pa_vs = g_main.vcount()

    coords_pa = np.array(g_main.vs['coords'])

    pa_vid_tree_starting_pts = np.where(np.array(g_main.vs['is_tree_startingPt_free']) == 1)[0]
    nr_of_tree_starting_pts = np.size(pa_vid_tree_starting_pts)                                # current node to connect

    print "Nr of remaining tree starting points:", nr_of_tree_starting_pts

    if nr_of_tree_starting_pts<1:
        return g_main, nr_of_tree_starting_pts

    current_pa_connecting_pt = pa_vid_tree_starting_pts[0]            # take current first open tree starting point
    coord_curr_tree_start_pt = coords_pa[current_pa_connecting_pt]    # coordinate of tree starting point

    print "Connecting tree to vertex", current_pa_connecting_pt, coord_curr_tree_start_pt

    # Read tree data and vertex offset to base graph
    adj_list_tree = np.array(g_daughter.get_edgelist(), dtype=np.int) + nr_pa_vs
    n_tree_vs = g_daughter.vcount()
    n_tree_es = g_daughter.ecount()
    coords_tree_moved = np.array(g_daughter.vs['coords']) + coord_curr_tree_start_pt  # transform tree to right location

    if type_daughter == 2:
        tree_vid_connect2pa = np.where(np.array(g_daughter.vs['is_connected_to_PA']) == 1)[0] + nr_pa_vs  # find node ids of DA which will be connected to PA
    elif type_daughter == 3:
        tree_vid_connect2pa = np.where(np.array(g_daughter.vs['is_connected_to_PV']) == 1)[0] + nr_pa_vs  # find node ids of AV which will be connected to PA
    else:
        print "Error: Invalid vessel type of daughter tree. Must be 2 (DA) or 3 (AV). Currently is", type_daughter
        sys.exit()

    if np.size(tree_vid_connect2pa) > 1:
        print "Error: more than one source node in DA (not implemented, currently...)"
        sys.exit()
    current_tree_connecting_pt = tree_vid_connect2pa[0]  # node id of tree which will be connected to PA

    adj_list_merged = np.append(adj_list_pa, adj_list_tree).reshape(-1, 2)  # merge adjacency lists of graphs
    coords_merged = np.append(coords_pa, coords_tree_moved).reshape(-1, 3)  # merge node coordinates

    es_merged_attr = {}  # prepare edge attributes of merged graph
    vs_merged_attr = {}  # prepare node attributes of merged graph

    vs_merged_attr['coords'] = coords_merged
    vs_merged_attr['ACA_in'] = np.append(g_main.vs['ACA_in'], np.zeros(n_tree_vs, dtype=np.int))
    vs_merged_attr['MCA_in'] = np.append(g_main.vs['MCA_in'], np.zeros(n_tree_vs, dtype=np.int))
    vs_merged_attr['COW_in'] = np.append(g_main.vs['COW_in'], np.zeros(n_tree_vs, dtype=np.int))
    vs_merged_attr['is_DA_startingPt'] = np.append(g_main.vs['is_DA_startingPt'], np.zeros(n_tree_vs, dtype=np.int))
    vs_merged_attr['is_DA_startingPt_added_manually'] = np.append(g_main.vs['is_DA_startingPt_added_manually'],np.zeros(n_tree_vs, dtype=np.int))
    vs_merged_attr['is_AV_root'] = np.append(g_main.vs['is_AV_root'], np.zeros(n_tree_vs, dtype=np.int))
    vs_merged_attr['is_connected_2caps'] = np.append(g_main.vs['is_connected_2caps'],g_daughter.vs['is_connected_2caps'])

    # set starting point to 0; is connected now
    vs_merged_attr['is_tree_startingPt_free'] = np.append(g_main.vs['is_tree_startingPt_free'], np.zeros(n_tree_vs, dtype=np.int))
    vs_merged_attr['is_tree_startingPt_free'][pa_vid_tree_starting_pts[0]] = 0  # if is connected, set to 0

    es_merged_attr['diameter'] = np.append(g_main.es['diameter'], g_daughter.es['diameter'])
    es_merged_attr['length'] = np.append(g_main.es['length'], g_daughter.es['length'])
    es_merged_attr['type'] = np.append(g_main.es['type'], g_daughter.es['type'])
    es_merged_attr['diam_post_exp'] = np.append(g_main.es['diam_post_exp'], -np.ones(n_tree_es))
    es_merged_attr['diam_pre_exp'] = np.append(g_main.es['diam_pre_exp'], -np.ones(n_tree_es))
    es_merged_attr['vRBC_post_exp'] = np.append(g_main.es['vRBC_post_exp'], np.array([None] * n_tree_es))
    es_merged_attr['vRBC_pre_exp'] = np.append(g_main.es['vRBC_pre_exp'], np.array([None] * n_tree_es))
    es_merged_attr['vRBC_pre_larger10'] = np.append(g_main.es['vRBC_pre_larger10'], np.zeros(n_tree_es, dtype=np.int))
    es_merged_attr['is_collateral'] = np.append(g_main.es['is_collateral'], np.zeros(n_tree_es, dtype=np.int))
    es_merged_attr['index_exp'] = np.append(g_main.es['index_exp'], -np.ones(n_tree_es, dtype=np.int))
    es_merged_attr['is_stroke'] = np.append(g_main.es['is_stroke'], g_daughter.es['is_stroke'])
    es_merged_attr['added_manually'] = np.append(g_main.es['added_manually'], np.ones(n_tree_es, dtype=np.int))

    # create new graph
    g_merged = igraph.Graph(adj_list_merged.tolist())

    for edge_attribute in es_merged_attr:
        g_merged.es[edge_attribute] = es_merged_attr[edge_attribute]

    for node_attribute in vs_merged_attr:
        g_merged.vs[node_attribute] = vs_merged_attr[node_attribute]

    for attribute in g_main.attributes():
        g_merged[attribute] = g_main[attribute]

    current_tree_connecting_pt_neighbours = g_merged.neighbors(current_tree_connecting_pt)  # find neighbours of tree head node
    if np.size(current_tree_connecting_pt_neighbours) > 1:
        print "Error: more than one neighbour for each current_da_connecting_pt (currently not implemented...)"
        sys.exit()

    eid_remove = g_merged.get_eid(current_tree_connecting_pt, current_tree_connecting_pt_neighbours[0])  # find eid of edge adjacent to head node

    g_merged.add_edge(current_pa_connecting_pt, current_tree_connecting_pt_neighbours[0],
                      diameter=g_merged.es['diameter'][eid_remove], length=g_merged.es['length'][eid_remove],
                      type=g_merged.es['type'][eid_remove], diam_post_exp=-1., diam_pre_exp=-1., index_exp=-1,
                      is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, vRBC_pre_larger10=0,
                      is_stroke=0, added_manually=1)  # add edge from PA to tree-neighbour of head node

    g_merged.delete_edges(eid_remove)  # delete remaining edge
    g_merged.delete_vertices(current_tree_connecting_pt)  # delete remaining node

    return g_merged, nr_of_tree_starting_pts


def add_av_trees_to_sa_graph(graph_parent, av_candidates, path_trees):

    nr_of_open_AV_starting_pts = 1e6

    graph_parent.vs['is_tree_startingPt_free'] = graph_parent.vs['is_AV_root']

    while nr_of_open_AV_starting_pts > 0:
        current_AV_id = np.random.choice(av_candidates)
        current_rotation = np.random.rand() * 2 * np.pi

        path_vein_es = path_trees + str(current_AV_id) + '_edgesDict.pkl'
        path_vein_vs = path_trees + str(current_AV_id) + '_verticesDict.pkl'

        g_av = import_av_graph_from_pkl(path_vein_es, path_vein_vs, rotation=current_rotation,
                                        l_scaling=2. / 3., min_diam=4.5, adj_attr_name='tuple', graph_name='B6 AV')

        print 'AV tree:', current_AV_id, 'Max. degree current AV:', np.max(g_av.degree())
        graph_parent, nr_of_open_AV_starting_pts = merge_sa_with_trees(graph_parent, g_av, 3)

    del graph_parent.vs['is_tree_startingPt_free']

    return graph_parent


def add_capillary_bed(g_parent, z_min_caps=25., frame_bounding_box=50., distance_between_cap_vs=45., l_vessel=62.,
                      d_vessel=4.5, perturb_vs_frac = 0.):

    # Find bounding box coordinates of capillary bed
    is_tree_connect_to_caps = np.in1d(np.array(g_parent.vs['is_connected_2caps']), 1)
    coords_connection_pts = np.array(g_parent.vs['coords'])[is_tree_connect_to_caps, :]

    x_min = np.min(coords_connection_pts[:, 0]) - frame_bounding_box
    x_max = np.max(coords_connection_pts[:, 0]) + frame_bounding_box

    y_min = np.min(coords_connection_pts[:, 1]) - frame_bounding_box
    y_max = np.max(coords_connection_pts[:, 1]) + frame_bounding_box

    z_min = z_min_caps
    z_max = np.max(coords_connection_pts[:, 2]) + frame_bounding_box

    print 'x_min, x_max, y_min, y_max, z_min, z_max:', x_min, x_max, y_min, y_max, z_min, z_max


    g_capbed = create_stacked_hex_network(x_min, x_max, y_min, y_max, z_min, z_max, distance_between_cap_vs, l_vessel,
                                          d_vessel, perturb_vs_frac=perturb_vs_frac)

    print 'Graph summary for cap bed', g_capbed.summary()

    g_merged = merge_with_capbed(g_parent, g_capbed)

    print 'Graph summary after merge PA with DAs and AVs and Caps', g_merged.summary()

    return g_merged


def merge_with_capbed(g_main, g_daughter):

    # Read main graph
    adj_list_main = np.array(g_main.get_edgelist(), dtype=np.int)
    nr_vs_main = g_main.vcount()
    coords_tree_nw = np.array(g_main.vs['coords'])

    # Read cap bed
    adj_list_caps = np.array(g_daughter.get_edgelist(), dtype=np.int) + nr_vs_main
    n_vs_caps = g_daughter.vcount()
    n_es_caps = g_daughter.ecount()
    coords_capbed = np.array(g_daughter.vs['coords'])

    adj_list_merged = np.append(adj_list_main, adj_list_caps).reshape(-1, 2)    # merge adjacency lists of graphs
    coords_merged = np.append(coords_tree_nw, coords_capbed).reshape(-1, 3)     # merge node coordinates

    es_merged_attr = {}     # prepare edge attributes of merged graph
    vs_merged_attr = {}     # prepare node attributes of merged graph

    vs_merged_attr['coords'] = coords_merged
    vs_merged_attr['ACA_in'] = np.append(g_main.vs['ACA_in'], np.zeros(n_vs_caps, dtype=np.int))
    vs_merged_attr['MCA_in'] = np.append(g_main.vs['MCA_in'], np.zeros(n_vs_caps, dtype=np.int))
    vs_merged_attr['COW_in'] = np.append(g_main.vs['COW_in'], np.zeros(n_vs_caps, dtype=np.int))
    vs_merged_attr['is_DA_startingPt'] = np.append(g_main.vs['is_DA_startingPt'], np.zeros(n_vs_caps, dtype=np.int))
    vs_merged_attr['is_DA_startingPt_added_manually'] = np.append(g_main.vs['is_DA_startingPt_added_manually'],
                                                                  np.zeros(n_vs_caps, dtype=np.int))
    vs_merged_attr['is_AV_root'] = np.append(g_main.vs['is_AV_root'], np.zeros(n_vs_caps, dtype=np.int))
    vs_merged_attr['is_connected_2caps'] = np.append(g_main.vs['is_connected_2caps'], np.zeros(n_vs_caps, dtype=np.int))

    es_merged_attr['diameter'] = np.append(g_main.es['diameter'], g_daughter.es['diameter'])
    es_merged_attr['length'] = np.append(g_main.es['length'], g_daughter.es['length'])
    es_merged_attr['type'] = np.append(g_main.es['type'], g_daughter.es['type'])
    es_merged_attr['diam_post_exp'] = np.append(g_main.es['diam_post_exp'], -np.ones(n_es_caps))
    es_merged_attr['diam_pre_exp'] = np.append(g_main.es['diam_pre_exp'], -np.ones(n_es_caps))
    es_merged_attr['vRBC_post_exp'] = np.append(g_main.es['vRBC_post_exp'],  np.array([None] * n_es_caps))
    es_merged_attr['vRBC_pre_exp'] = np.append(g_main.es['vRBC_pre_exp'],  np.array([None] * n_es_caps))
    es_merged_attr['vRBC_pre_larger10'] = np.append(g_main.es['vRBC_pre_larger10'], np.zeros(n_es_caps, dtype=np.int))
    es_merged_attr['is_collateral'] = np.append(g_main.es['is_collateral'], np.zeros(n_es_caps, dtype=np.int))
    es_merged_attr['index_exp'] = np.append(g_main.es['index_exp'], -np.ones(n_es_caps, dtype=np.int))
    es_merged_attr['is_stroke'] = np.append(g_main.es['is_stroke'], np.zeros(n_es_caps, dtype=np.int))
    es_merged_attr['added_manually'] = np.append(g_main.es['added_manually'], np.ones(n_es_caps, dtype=np.int))

    # create new graph
    g_merged = igraph.Graph(adj_list_merged.tolist())

    for edge_attribute in es_merged_attr:
        g_merged.es[edge_attribute] = es_merged_attr[edge_attribute]

    for node_attribute in vs_merged_attr:
        g_merged.vs[node_attribute] = vs_merged_attr[node_attribute]

    for attribute in g_main.attributes():
        g_merged[attribute] = g_main[attribute]

    g_merged.vs['is_connection_to_caps_free'] = g_merged.vs['is_connected_2caps']

    # Loop for connecting caps to DA and AV trees
    vs_ids_connected2Bed = np.where(np.array(g_merged.vs['is_connection_to_caps_free']) == 1)[0]
    nr_of_remaining_connections = np.size(vs_ids_connected2Bed)
    print 'Nr of remaining connections from capillary bed to trees', np.size(vs_ids_connected2Bed)
    print 'Connecting... (implementation for connecting caps with trees a bit slow, should be optimized...)'

    while nr_of_remaining_connections>0:

        if nr_of_remaining_connections % 100==0:
            print 'Nr of remaining connections from Capbed to trees', nr_of_remaining_connections

        current_vs_id_connection = vs_ids_connected2Bed[0]
        coords_vs_connect = np.array(g_merged.vs['coords'])[current_vs_id_connection]

        adj_list_merged = np.array(g_merged.get_edgelist(), dtype=np.int)
        coords_edge_midpts = (np.array(g_merged.vs['coords'])[adj_list_merged[:,0]]+np.array(g_merged.vs['coords'])[adj_list_merged[:,1]])/2.

        distance = np.linalg.norm(np.array([coords_edge_midpts[:,0]-coords_vs_connect[0], coords_edge_midpts[:,1]-coords_vs_connect[1], coords_edge_midpts[:,2]-coords_vs_connect[2]]),axis=0)
        type = np.array(g_merged.es['type'])
        distance[np.logical_not(np.in1d(type,4))] = 10*np.max(distance) # only include capillaries as connection canditates

        es_id_closest = np.argmin(distance)
        length_e_close = np.array(g_merged.es['length'])[es_id_closest]
        diamet_e_close = np.array(g_merged.es['diameter'])[es_id_closest]
        type_e_closest = type[es_id_closest]
        if type_e_closest < 4:
            print 'Error: Something wrong.. cannot connect to other edge than cap..'
            sys.exit()

        vs_l = adj_list_merged[es_id_closest, 0]
        vs_r = adj_list_merged[es_id_closest, 1]

        g_merged.add_edge(vs_l, current_vs_id_connection, diameter=diamet_e_close, length=length_e_close,
                          type=type_e_closest, diam_post_exp=-1., diam_pre_exp=-1., index_exp=-1, is_collateral=0,
                          vRBC_post_exp=None, vRBC_pre_exp=None, vRBC_pre_larger10=0, is_stroke=0, added_manually=1)  # assign same length to each edge
        g_merged.add_edge(current_vs_id_connection, vs_r, diameter=diamet_e_close, length=length_e_close,
                          type=type_e_closest, diam_post_exp=-1., diam_pre_exp=-1., index_exp=-1, is_collateral=0,
                          vRBC_post_exp=None, vRBC_pre_exp=None, vRBC_pre_larger10=0, is_stroke=0,added_manually=1)
        g_merged.delete_edges(es_id_closest)  # delete old edge

        is_free_connection_pt = np.array(g_merged.vs['is_connection_to_caps_free'])
        is_free_connection_pt[current_vs_id_connection] = 0
        g_merged.vs['is_connection_to_caps_free'] = is_free_connection_pt

        vs_ids_connected2Bed = np.where(np.array(g_merged.vs['is_connection_to_caps_free']) == 1)[0]
        nr_of_remaining_connections = np.size(vs_ids_connected2Bed)

    print 'Nr of remaining connections from capillary bed to trees', np.size(vs_ids_connected2Bed)
    print 'All caps connected'

    del g_merged.vs['is_connection_to_caps_free'] # delete temporary attribute

    return g_merged


def remove_regions_outside_DAs(graph, max_distance_to_da):

    # nkind = 4 all honeycomb capillaries
    # nkind = 3 all added AVs
    # nkind = 2 all added DAs
    # nkind = 0 pial vessels

    # nkind is type for vertices

    nkind = np.ones(graph.vcount(), dtype=np.int) * 4

    vs_indices_all = np.arange(graph.vcount())
    adjacency_list = np.array(graph.get_edgelist(), dtype=np.int)
    vessel_type = np.array(graph.es['type'])

    is_AV = np.in1d(vessel_type, 3)
    nodes_AV = np.unique(adjacency_list[is_AV])
    is_node_av = np.in1d(vs_indices_all, nodes_AV)
    nkind[is_node_av] = 3

    is_PA = np.logical_or(np.in1d(vessel_type, 0), np.in1d(vessel_type, -1))
    nodes_PA = np.unique(adjacency_list[is_PA])
    is_node_pa = np.in1d(vs_indices_all, nodes_PA)
    nkind[is_node_pa] = 0

    is_DA = np.in1d(vessel_type, 2)
    nodes_DA = np.unique(adjacency_list[is_DA])
    is_node_da = np.in1d(vs_indices_all, nodes_DA)
    nkind[is_node_da] = 2

    graph.vs['nkind'] = nkind
    graph.vs['global_index'] = vs_indices_all

    tree_DAs = KDTree(graph.vs(nkind_eq=2)['coords'])
    allDists, vertex_indices_local = tree_DAs.query(np.array(graph.vs(nkind_ge=3)['coords']))

    # delete all AVs which are at certain distance to DAs
    vertex_indices_global = np.array(graph.vs(nkind_ge=3)['global_index'])
    verticesToDelete = vertex_indices_global[allDists > max_distance_to_da]
    graph.delete_vertices(verticesToDelete)

    # Delete resulting capillary and AV dead Ends
    graph.vs['degree'] = graph.degree()
    while len(graph.vs(degree_eq=1, nkind_ge=3, is_AV_root_eq=0)) != 0:  # AND condition. delete vertices with degree 1, are caps or AVs and are not a AV root point
        verticesToDelete = graph.vs(degree_eq=1, nkind_ge=3, is_AV_root_eq=0).indices
        graph.delete_vertices(verticesToDelete)
        graph.vs['degree'] = graph.degree()

    # delete degree 0 vertices (vertices that are now completely disconnected)
    graph.vs['degree'] = graph.degree()
    verticesToDelete_0 = graph.vs(degree_eq=0, nkind_ge=3).indices
    graph.delete_vertices(verticesToDelete_0)

    del graph.vs['degree'], graph.vs['nkind'], graph.vs['global_index']

    print 'Network cropped'

    return graph