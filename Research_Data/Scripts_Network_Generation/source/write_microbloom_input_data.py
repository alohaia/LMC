import csv
import numpy as np


def export_node_data(x, y, z, filepath = "node_data.csv", delimiter =','):

    export_matrix_nodes = np.zeros((np.size(x), 3))

    export_matrix_nodes[:,0] = x
    export_matrix_nodes[:,1] = y
    export_matrix_nodes[:,2] = z

    header = [['x','y','z']]
    with open(filepath, "wb") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(header)
        writer.writerows(export_matrix_nodes)

    return None


def export_edge_data(edge_list, diameter, length, filepath = "edge_data.csv", delimiter =','):

    nr_of_edges = np.size(edge_list,0)

    export_matrix_edges = np.zeros((nr_of_edges, 4))
    export_matrix_edges[:, 0:2] = edge_list
    export_matrix_edges[:, 2] = diameter
    export_matrix_edges[:, 3] = length

    header = [['n1', 'n2', 'D', 'L']]
    with open(filepath, "wb") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(header)
        writer.writerows(export_matrix_edges)

    return None


def export_boundary_node_data_dirichletAndNeuman(bc_vs, bc_type, bc_press, bc_flux, filepath="boundary_data_nodes.csv", delimiter =','):

    export_matrix = np.zeros((np.size(bc_vs),4))

    export_matrix[:, 0] = bc_vs
    export_matrix[:, 1] = bc_type
    export_matrix[:, 2] = bc_press
    export_matrix[:, 3] = bc_flux

    header = [['nodeId', 'boundaryType', 'p', 'flux']]
    with open(filepath, "wb") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(header)
        writer.writerows(export_matrix)
    return None
