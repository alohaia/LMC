import numpy as np
import igraph


def create_stacked_hex_network(x_min, x_max, y_min, y_max, z_min, z_max, dist_between_nodes, l_vessel, d_vessel, perturb_vs_frac=0.):
    dx_box = x_max - x_min
    dy_box = y_max - y_min
    dz_box = z_max - z_min

    # generate honey comb
    x_cap_vs_hexa, y_cap_vs_hexa, adj_edgelist_caps_hexa = __get_2d_honeycomb(dx_box, dy_box, 2 * dist_between_nodes)

    # split honey comb
    x_caps_vs_splited, y_caps_vs_splited, adj_edgelist_splited,is_split_node = __split_honeycomb(x_cap_vs_hexa, y_cap_vs_hexa, adj_edgelist_caps_hexa)

    # stack honeycomb, vertical connections between stacks
    x_cap_stacked,y_cap_stacked,z_cap_stacked,adj_edgelist_stacked = __stack_honeycomb(x_caps_vs_splited, y_caps_vs_splited, adj_edgelist_splited,is_split_node, dist_between_nodes, dz_box)

    x_cap_stacked += x_min
    y_cap_stacked += y_min
    z_cap_stacked += z_min

    nr_of_es = np.size(adj_edgelist_stacked,0)
    nr_of_vs = np.size(x_cap_stacked)

    diameter = np.ones(nr_of_es) * d_vessel
    length = np.ones(nr_of_es) * l_vessel
    type = np.ones(nr_of_es, dtype=int) * 4             # 0-PA; 1-PV; 2-DA; 3-AV; 4-C

    if perturb_vs_frac>0:
        x_cap_stacked += ((np.random.rand(nr_of_vs) - .5) * dist_between_nodes * perturb_vs_frac)
        y_cap_stacked += ((np.random.rand(nr_of_vs) - .5) * dist_between_nodes * perturb_vs_frac)
        z_cap_stacked += ((np.random.rand(nr_of_vs) - .5) * dist_between_nodes * perturb_vs_frac)

    coords = np.zeros((nr_of_vs,3))
    coords[:,0] = x_cap_stacked
    coords[:, 1] = y_cap_stacked
    coords[:, 2] = z_cap_stacked

    g = igraph.Graph(adj_edgelist_stacked.tolist())
    g.es['diameter'] = diameter
    g.es['length'] = length
    g.es['type'] = type
    g.vs['coords'] = coords

    return g


def __get_2d_honeycomb(dx_box, dy_box, l_doub_edge):

    dx_cell = 1.5 * l_doub_edge
    dy_cell = np.sqrt(3.) * l_doub_edge

    nr_of_cells_x = int(np.ceil(dx_box / dx_cell))
    nr_of_cells_y = int(np.ceil(dy_box / dy_cell))

    if nr_of_cells_x % 2 == 0:
        nr_of_cells_x += 1
        print('Nr of hexagon in x direction increased to get odd number...')
    if nr_of_cells_y % 2 == 0:
        nr_of_cells_y += 1
        print('Nr hexagon in y direction increased to get odd number...')

    print('Nr of hexagon in x', nr_of_cells_x)
    print('Nr of hexagon in y', nr_of_cells_y)

    n_vs_x = nr_of_cells_x + 1

    n_vs_y = nr_of_cells_y * 2 + 1
    n_vs_xy = n_vs_x * n_vs_y

    x = np.zeros(n_vs_xy)
    y = np.zeros(n_vs_xy)

    for ii in range(n_vs_xy):

        i = ii % n_vs_x
        j = ii // n_vs_x

        y[ii] = j * l_doub_edge / 2 * np.sqrt(3)

        if j % 2 == 0:
            if i == 0:
                x[ii] = l_doub_edge / 2
            elif i % 2 == 1:
                x[ii] = x[ii - 1] + l_doub_edge
            else:
                x[ii] = x[ii - 1] + 2 * l_doub_edge
        else:
            if i == 0:
                x[ii] = 0
            elif i % 2 == 1:
                x[ii] = x[ii - 1] + 2 * l_doub_edge
            else:
                x[ii] = x[ii - 1] + l_doub_edge

    nr_of_edges = n_vs_x / 2 * (3 * (n_vs_y - 1) + 1) - (n_vs_y - 1) / 2
    adj_edgelist_hexa = np.ones((round(nr_of_edges), 2), dtype=int) * (-1)

    # create edges
    eid = 0

    for ii in range(n_vs_xy):
        i = ii % n_vs_x
        j = ii // n_vs_x
        if (j % 2 == 0 and i % 2 == 0 and i < n_vs_x - 1) or (j % 2 == 1 and i % 2 == 1 and i < n_vs_x - 2):
            adj_edgelist_hexa[eid, 0] = ii
            adj_edgelist_hexa[eid, 1] = ii + 1
            eid += 1

    for ii in range(n_vs_xy):
        i = ii % n_vs_x
        j = ii // n_vs_x
        ii_tr = ii + n_vs_x
        ii_br = ii - n_vs_x
        if (i % 2 == 1 and j % 2 == 0) or (i % 2 == 0 and j % 2 == 1):
            if ii_tr < n_vs_xy:
                adj_edgelist_hexa[eid, 0] = ii
                adj_edgelist_hexa[eid, 1] = ii_tr
                eid += 1
            if ii_br > -1:
                adj_edgelist_hexa[eid, 0] = ii
                adj_edgelist_hexa[eid, 1] = ii_br
                eid += 1

    print(np.max(x))

    return x,y,adj_edgelist_hexa

def __split_honeycomb(x_cap_vs_hexa, y_cap_vs_hexa, adj_edgelist_caps_hexa):

    nr_of_edges_hexa = np.size(adj_edgelist_caps_hexa, 0)
    nr_of_nodes_hexa = np.size(x_cap_vs_hexa, 0)

    nr_of_split_nodes = nr_of_edges_hexa

    x_split_node = np.sum(x_cap_vs_hexa[adj_edgelist_caps_hexa], 1) / 2.
    y_split_node = np.sum(y_cap_vs_hexa[adj_edgelist_caps_hexa], 1) / 2.
    node_ids_split_nodes = np.arange(nr_of_split_nodes) + nr_of_nodes_hexa  # higher ids

    adj_edgelist_caps_I = np.copy(adj_edgelist_caps_hexa)
    adj_edgelist_caps_II = np.copy(adj_edgelist_caps_hexa)

    adj_edgelist_caps_I[:, 1] = node_ids_split_nodes
    adj_edgelist_caps_II[:, 0] = node_ids_split_nodes

    adj_edgelist_splited = np.append(adj_edgelist_caps_I, adj_edgelist_caps_II).reshape(-1, 2)

    x_caps_vs_splited = np.append(x_cap_vs_hexa, x_split_node)
    y_caps_vs_splited = np.append(y_cap_vs_hexa, y_split_node)

    is_split_node = np.zeros(nr_of_nodes_hexa+nr_of_split_nodes)
    is_split_node[node_ids_split_nodes] = 1

    return x_caps_vs_splited,y_caps_vs_splited,adj_edgelist_splited,is_split_node


def __stack_honeycomb(x_caps_vs_splited, y_caps_vs_splited, adj_edgelist_splited,is_split_node, l_edge, dz_box):

    node_ids = np.arange(np.size(x_caps_vs_splited))

    dz_cell = l_edge
    nr_of_layers = int(np.ceil(dz_box / dz_cell))

    split_nodes = node_ids[np.isin(is_split_node, 1)]
    nr_of_split_nodes = np.size(split_nodes)

    vertical_connection_nodes_layer_I = np.sort(np.random.choice(split_nodes, nr_of_split_nodes // 2, replace=False))
    vertical_connection_nodes_layer_II = split_nodes[np.logical_not(np.isin(split_nodes, vertical_connection_nodes_layer_I))]

    nr_of_vs_per_layer = np.size(x_caps_vs_splited)

    # initialize first layer
    x_cap_stacked = np.copy(x_caps_vs_splited)
    y_cap_stacked = np.copy(y_caps_vs_splited)
    z_cap_stacked = np.zeros(nr_of_vs_per_layer)
    adj_edgelist_stacked = np.copy(adj_edgelist_splited)

    # add other layers
    for current_layer in np.arange((nr_of_layers - 1)) + 1:
        current_z = l_edge * current_layer

        x_cap_stacked = np.append(x_cap_stacked, x_caps_vs_splited)
        y_cap_stacked = np.append(y_cap_stacked, y_caps_vs_splited)
        z_cap_stacked = np.append(z_cap_stacked, np.ones(nr_of_vs_per_layer) * current_z)

        adj_edgelist_stacked = np.append(adj_edgelist_stacked,
                                         adj_edgelist_splited + current_layer * nr_of_vs_per_layer).reshape(-1, 2)

    # add vertical connections
    for current_layer in np.arange((nr_of_layers - 1)) + 1:

        if current_layer % 2 == 0:
            nr_of_vertical_edges_current = np.size(vertical_connection_nodes_layer_I)
            edge_list_vertical = -np.ones((nr_of_vertical_edges_current, 2), dtype=int)
            edge_list_vertical[:, 0] = vertical_connection_nodes_layer_I + (current_layer - 1) * nr_of_vs_per_layer
            edge_list_vertical[:, 1] = vertical_connection_nodes_layer_I + (current_layer) * nr_of_vs_per_layer
        else:
            nr_of_vertical_edges_current = np.size(vertical_connection_nodes_layer_II)
            edge_list_vertical = -np.ones((nr_of_vertical_edges_current, 2), dtype=int)
            edge_list_vertical[:, 0] = vertical_connection_nodes_layer_II + (current_layer - 1) * nr_of_vs_per_layer
            edge_list_vertical[:, 1] = vertical_connection_nodes_layer_II + (current_layer) * nr_of_vs_per_layer

        adj_edgelist_stacked = np.append(adj_edgelist_stacked, edge_list_vertical).reshape(-1, 2)

    return x_cap_stacked,y_cap_stacked,z_cap_stacked,adj_edgelist_stacked
