import numpy as np
import igraph
import pickle
import sys
import time
import datetime


def import_sa_graph_from_pkl(path_es_dict, path_vs_dict, adj_attr_name ='connectivity', graph_name='tmp', delete_tortiosity = True):

    with open(path_es_dict, 'rb') as f:
        data_edge = pickle.load(f)
    with open(path_vs_dict, 'rb') as f:
        data_vertex = pickle.load(f)

    adjlist = np.array(data_edge[adj_attr_name])

    g = igraph.Graph(adjlist.tolist())

    for edge_attribute in data_edge:
        g.es[edge_attribute] = data_edge[edge_attribute]

    for node_attribute in data_vertex:
        g.vs[node_attribute] = data_vertex[node_attribute]

    ts = time.time()
    g['network_gen_date'] = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    g['Mouse'] = graph_name

    g.es['type'] = np.zeros(g.ecount(), dtype=int)     # 0-PA; 1-PV; 2-DA; 3-AV; 4-C

    is_aca_in = np.isin(np.array(g.vs['ACA_in']),1)
    is_mca_in = np.isin(np.array(g.vs['MCA_in']),1)
    is_degree1 = np.isin(np.array(g.vs['degree']),1)

    # fix an issue, where None was used in Attribute ACA_in and MCA_in, instead of 0
    del g.vs['ACA_in'], g.vs['MCA_in']
    aca_in = np.zeros(g.vcount(), dtype=int)
    aca_in[is_aca_in] = 1
    mca_in = np.zeros(g.vcount(), dtype=int)
    mca_in[is_mca_in] = 1
    g.vs['ACA_in'] = aca_in
    g.vs['MCA_in'] = mca_in

    # find vs connected to DA (DA starting points)
    is_DA_startingPt = np.logical_and(is_degree1, np.logical_not(np.logical_or(is_aca_in, is_mca_in)))

    DA_startingPt = np.zeros(g.vcount(),dtype=int)
    DA_startingPt[is_DA_startingPt] = 1

    g.vs['is_DA_startingPt'] = DA_startingPt
    g.vs['is_DA_startingPt_added_manually'] = np.zeros(g.vcount(),dtype=int)
    g.vs['is_AV_root'] = np.zeros(g.vcount(), dtype=int)
    g.vs['is_connected_2caps'] = np.zeros(g.vcount(), dtype=int)
    g.vs['COW_in'] = np.zeros(g.vcount(),dtype=int)

    g.es['is_stroke'] = np.zeros(g.ecount(), dtype=int)

    del g.vs['degree'], g.vs['index']
    del g.es['connectivity']

    if delete_tortiosity:
        del g.es['lengths2'], g.es['points'], g.es['diameters']

    return g


def check_graph_attributes_complete_nwgen(graph):

    current_graph_attributes = graph.attributes()
    current_vertex_attributes = graph.vs.attributes()
    current_edge_attributes = graph.es.attributes()

    target_attributes = ['Mouse', 'network_gen_date']

    target_vert_attri = ['ACA_in', 'MCA_in', 'COW_in', 'coords', 'is_AV_root', 'is_DA_startingPt',
                         'is_DA_startingPt_added_manually', 'is_connected_2caps']

    target_edge_attri = ['added_manually', 'diam_post_exp', 'diam_pre_exp', 'diameter', 'index_exp', 'is_collateral',
                         'is_stroke', 'length', 'type', 'vRBC_post_exp', 'vRBC_pre_exp', 'vRBC_pre_larger10']

    # deleted:  'diam_post0_exp', 'diam_post30_exp', 'diam_post60_exp', 'diam_post90_exp', 'diam_post120_exp',
    #           'vRBC_post0_exp', 'vRBC_post0_larger10', 'vRBC_post30_exp', 'vRBC_post30_larger10', 'vRBC_post60_exp',
    #           'vRBC_post60_larger10', 'vRBC_post90_exp',     'vRBC_post90_larger10', 'vRBC_post120_exp',
    #           'vRBC_post120_larger10',

    attributes_missing_in_graph = []
    attributes_excessive_in_graph = []
    is_consistent = True

    if len(set(target_attributes)) != len(target_attributes) or len(set(target_vert_attri)) != len(
            target_vert_attri) or len(set(target_edge_attri)) != len(target_edge_attri):
        is_consistent = False
        print("List of target edge attributes not unique.")
        return is_consistent

    # check if all target attributes are in current graph
    # Graph attributes
    for attr in target_attributes:
        if attr in current_graph_attributes:
            continue
        else:
            attributes_missing_in_graph.append(attr)
            is_consistent = False

    # Vertex attributes
    for attr in target_vert_attri:
        if attr in current_vertex_attributes:
            continue
        else:
            attributes_missing_in_graph.append(attr)
            is_consistent = False

    # Edge attributes
    for attr in target_edge_attri:
        if attr in current_edge_attributes:
            continue
        else:
            attributes_missing_in_graph.append(attr)
            is_consistent = False

    # Check if no additional attributes are in current graph
    # Graph attributes
    for attr in current_graph_attributes:
        if attr in target_attributes:
            continue
        else:
            attributes_excessive_in_graph.append(attr)
            is_consistent = False

    # Vertex attributes
    for attr in current_vertex_attributes:
        if attr in target_vert_attri:
            continue
        else:
            attributes_excessive_in_graph.append(attr)
            is_consistent = False

    # Edge attributes
    for attr in current_edge_attributes:
        if attr in target_edge_attri:
            continue
        else:
            attributes_excessive_in_graph.append(attr)
            is_consistent = False


    if not is_consistent:
        print("Something is wrong with graph attributes: ")
        print("The following attributes are missing in current graph:", attributes_missing_in_graph)
        print("The following attributes are excessive in current graph:", attributes_excessive_in_graph)

    return is_consistent


def import_da_av_graph_from_pkl(path_es_dict, path_vs_dict, type, l_scaling = 1., min_diam = 6., rotation = 0,
                                adj_attr_name ='connectivity', graph_name='tmp'):

    # type 2: DA, 3: AV

    with open(path_es_dict, 'rb') as f:
        data_edge = pickle.load(f)
    with open(path_vs_dict, 'rb') as f:
        data_vertex = pickle.load(f)
    adjlist = np.array(data_edge[adj_attr_name])

    g = igraph.Graph(adjlist.tolist())

    for edge_attribute in data_edge:
        g.es[edge_attribute] = data_edge[edge_attribute]

    for node_attribute in data_vertex:
        g.vs[node_attribute] = data_vertex[node_attribute]

    g['Mouse'] = graph_name

    length = np.zeros(g.ecount(), dtype=np.double)

    for e_id in range(g.ecount()):
        nr_subsegments = np.size(g.es['points'][e_id],0) - 1
        for sub_edge in range(nr_subsegments):
            node_coord_l = g.es['points'][e_id][sub_edge]
            node_coord_r = g.es['points'][e_id][sub_edge+1]
            dx = node_coord_l[0] - node_coord_r[0]
            dy = node_coord_l[1] - node_coord_r[1]
            dz = node_coord_l[2] - node_coord_r[2]
            length_sub_es = np.linalg.norm(np.array([dx,dy,dz])) * l_scaling
            length[e_id] += length_sub_es


    diameter = np.array(g.es['diameter'])
    diameter[diameter<min_diam] = min_diam

    del g.es['diameter']

    g['Mouse'] = graph_name
    g.es['diameter'] = diameter
    g.es['length'] = length
    g.es['is_stroke'] = np.zeros(g.ecount(), dtype=int)
    g.es['type'] = np.ones(g.ecount(), dtype=int) * type  # 0-PA; 1-PV; 2-DA; 3-AV; 4-C

    degree = np.array(g.degree())

    if type == 2:
        g.vs['is_connected_to_PA'] = g.vs['attachmentVertex']
        g.vs['is_connected_to_PV'] = np.zeros(g.vcount())
        is_connected_to_caps = np.logical_and(np.isin(degree, 1),
                                              np.logical_not(np.isin(np.array(g.vs['is_connected_to_PA']), 1)))

        is_da_root_vs = np.isin(np.array(g.vs['is_connected_to_PA']), 1)
        coord_tree_root = np.array(g.vs['coords'])[is_da_root_vs][0]
        if np.size(coord_tree_root) != 3:
            print("Error: More than one attachment vertex")
            sys.exit()

    elif type == 3:
        g.vs['is_connected_to_PV'] = g.vs['attachmentVertex']
        g.vs['is_connected_to_PA'] = np.zeros(g.vcount())
        is_connected_to_caps = np.logical_and(np.isin(degree, 1),
                                              np.logical_not(np.isin(np.array(g.vs['is_connected_to_PV']), 1)))

        is_av_outflow_vs = np.isin(np.array(g.vs['is_connected_to_PV']), 1)
        coord_tree_root = np.array(g.vs['coords'])[is_av_outflow_vs][0]
        if np.size(coord_tree_root) != 3:
            print("Error: More than one venule outflow vertex")
            sys.exit()

    else:
        print("Vessel type is not valid. Must be 2 or 3. Currently is", type)
        sys.exit()

    connected_to_caps = np.zeros(g.vcount(),dtype=int)
    connected_to_caps[is_connected_to_caps] = 1
    g.vs['is_connected_2caps'] = connected_to_caps

    vs_coord = np.array(g.vs['coords'])
    vs_coord_scaled = (vs_coord - coord_tree_root)*l_scaling

    rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0.],
                           [np.sin(rotation),  np.cos(rotation), 0.],
                           [0.              , 0.               , 1.]])

    vs_coord_rotated = np.transpose(np.dot(rot_matrix, np.transpose(vs_coord_scaled)))

    del g.vs['coords']
    g.vs['coords'] = vs_coord_rotated

    del g.es['points'], g.es['tuple']
    del g.vs['attachmentVertex']

    return g


def import_da_graph_from_pkl(path_es_dict, path_vs_dict, l_scaling = 1., min_diam = 6., rotation = 0., adj_attr_name ='connectivity', graph_name='tmp'):

    # previously, there was a separate implementation for importing AVs and DAs. See git branch 4.3.21 and earlier
    return  import_da_av_graph_from_pkl(path_es_dict, path_vs_dict, 2, l_scaling=l_scaling, min_diam=min_diam,
                                        rotation=rotation, adj_attr_name=adj_attr_name, graph_name=graph_name)


def import_av_graph_from_pkl(path_es_dict, path_vs_dict, l_scaling = 1., min_diam = 6., rotation = 0., adj_attr_name ='connectivity', graph_name='tmp'):

    # previously, there was a separate implementation for importing AVs and DAs. See git branch 4.3.21 and earlier
    return import_da_av_graph_from_pkl(path_es_dict, path_vs_dict, 3, l_scaling=l_scaling, min_diam=min_diam,
                                        rotation=rotation, adj_attr_name=adj_attr_name, graph_name=graph_name)
