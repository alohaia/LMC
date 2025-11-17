import numpy as np
import csv
from copy import deepcopy

from write_microbloom_input_data import export_node_data, export_edge_data, export_boundary_node_data_dirichletAndNeuman


def write_vtp(graph, filename, verbose=False, coordinatesKey='r'):
    """Writes a graph in iGraph format to a vtp-file (e.g. for plotting with
    Paraview). Adds an index to both edges and vertices to make comparisons
    with the iGraph format easier.
    INPUT: graph: Graph in iGraph format
           filename: Name of the vtp-file to be written. Note that no filename-
                     ending is appended automatically.
           verbose: Whether or not to print to the screen if writing an array
                    fails.Default is False
           coordinatesKey: Key of vertex attribute in which the coordinates are stored. Default is 'r'.
    OUTPUT: vtp-file written to disk.
    """

    # Make a copy of the graph so that modifications are possible, whithout
    # changing the original. Add indices that can be used for comparison with
    # the original, even after some edges / vertices in the copy have been
    # deleted:
    G = deepcopy(graph)
    G.vs['index'] = range(G.vcount())
    if G.ecount() > 0:
        G.es['index'] = range(G.ecount())

    # Delete selfloops as they cannot be viewed as straight cylinders and their
    # 'angle' property is 'nan':
    G.delete_edges(np.nonzero(G.is_loop())[0].tolist())

    tab = "  "
    fname = filename
    f = open(fname, 'w')

    # Find unconnected vertices:
    unconnected = np.nonzero([x == 0 for x in G.strength(weights=
                                                         [1 for i in range(G.ecount())])])[0].tolist()

    # Header
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="PolyData" version="0.1" ')
    f.write('byte_order="LittleEndian">\n')
    f.write('{}<PolyData>\n'.format(tab))
    f.write('{}<Piece NumberOfPoints="{}" '.format(2 * tab, G.vcount()))
    f.write('NumberOfVerts="{}" '.format(len(unconnected)))
    f.write('NumberOfLines="{}" '.format(G.ecount()))
    f.write('NumberOfStrips="0" NumberOfPolys="0">\n')

    # Vertex data
    keys = G.vs.attribute_names()
    keysToRemove = ['r', 'pBC', 'rBC', 'kind', 'sBC', 'inflowE', 'outflowE', 'adjacent', 'mLocation', 'lDir',
                    'diameter']
    for key in keysToRemove:
        if key in keys:
            keys.remove(key)
    f.write('{}<PointData Scalars="Scalars_p">\n'.format(3 * tab))
    for key in keys:
        write_array(f, G.vs[key], key, verbose=verbose)
    f.write('{}</PointData>\n'.format(3 * tab))

    # Edge data
    keys = G.es.attribute_names()
    keysToRemove = ['diameters', 'lengths', 'points', 'rRBC', 'tRBC']
    for key in keysToRemove:
        if key in keys:
            keys.remove(key)
    f.write('{}<CellData Scalars="diameter">\n'.format(3 * tab))
    for key in keys:
        write_array(f, G.es[key], key, zeros=len(unconnected), verbose=verbose)
    f.write('{}</CellData>\n'.format(3 * tab))

    # Vertices
    f.write('{}<Points>\n'.format(3 * tab))
    write_array(f, np.vstack(G.vs[coordinatesKey]), coordinatesKey, verbose=verbose)
    f.write('{}</Points>\n'.format(3 * tab))

    # Unconnected vertices
    if unconnected != []:
        f.write('{}<Verts>\n'.format(3 * tab))
        f.write('{}<DataArray type="Int64" '.format(4 * tab))
        f.write('Name="connectivity" format="ascii">\n')
        for vertex in unconnected:
            f.write('{}{}\n'.format(5 * tab, vertex))
        f.write('{}</DataArray>\n'.format(4 * tab))
        f.write('{}<DataArray type="Int64" '.format(4 * tab))
        f.write('Name="offsets" format="ascii">\n')
        for i in range(len(unconnected)):
            f.write('{}{}\n'.format(5 * tab, 1 + i))
        f.write('{}</DataArray>\n'.format(4 * tab))
        f.write('{}</Verts>\n'.format(3 * tab))

        # Edges
    f.write('{}<Lines>\n'.format(3 * tab))
    f.write('{}<DataArray type="Int64" '.format(4 * tab))
    f.write('Name="connectivity" format="ascii">\n')
    for edge in G.get_edgelist():
        f.write('{}{} {}\n'.format(5 * tab, edge[0], edge[1]))
    f.write('{}</DataArray>\n'.format(4 * tab))
    f.write('{}<DataArray type="Int64" '.format(4 * tab))
    f.write('Name="offsets" format="ascii">\n')
    for i in range(G.ecount()):
        f.write('{}{}\n'.format(5 * tab, 2 + i * 2))
    f.write('{}</DataArray>\n'.format(4 * tab))
    f.write('{}</Lines>\n'.format(3 * tab))

    # Footer
    f.write('{}</Piece>\n'.format(2 * tab))
    f.write('{}</PolyData>\n'.format(1 * tab))
    f.write('</VTKFile>\n')

    f.close()


def write_array(f, array, name, zeros=0, verbose=False):
    """Print arrays with different number of components, setting NaNs to 'substitute'.
    Optionally, a given number of zero-entries can be prepended to an
    array. This is required when the graph contains unconnected vertices.
    """
    tab = "  ";
    space = 5 * tab
    substituteD = -1000.;
    substituteI = -1000
    zeroD = 0.;
    zeroI = 0
    try:  # For arrays where attributes are vectors (e.g. coordinates)
        noc = np.shape(array)[1];
        firstel = array[0][0]
        Nai = len(array);
        Naj = np.array(map(len, array), dtype='int')

    except:
        noc = 1;
        firstel = array[0]
        Nai = len(array);
        Naj = np.array([0], dtype='int')

    if type(firstel) == str:
        if verbose:
            print("WARNING: array '%s' contains data of type 'string'!" % name)
        return  # Cannot have string-representations in paraview.
    if "<type 'NoneType'>" in map(str, np.unique(np.array(map(type, array)))):
        if verbose:
            print("WARNING: array '%s' contains data of type 'None'!" % name)
        return
    if any([type(firstel) == x for x in
            [float, np.float32, np.float64, np.float128]]):
        atype = "Float64"
        format = "%f"
    elif any([type(firstel) == x for x in
              [int, np.int8, np.int16, np.int32, np.int64]]):
        atype = "Int64"
        format = "%i"
    else:
        if verbose:
            print("WARNING: array '%s' contains data of unknown type!" % name)
        return

    f.write('{}<DataArray type="{}" Name="{}" '.format(4 * tab, atype, name))
    f.write('NumberOfComponents="{}" format="ascii">\n'.format(noc))

    if noc == 1:
        if atype == "Float64":
            for i in range(zeros):
                f.write('{}{}\n'.format(space, zeroD))
            aoD = np.array(array, dtype='double')
            for i in range(Nai):
                if not np.isfinite(aoD[i]):
                    f.write('{}{}\n'.format(space, substituteD))
                else:
                    f.write('{}{}\n'.format(space, aoD[i]))
        elif atype == "Int64":
            for i in range(zeros):
                f.write('{}{}\n'.format(space, zeroI))
            aoI = np.array(array, dtype=np.int64)
            for i in range(Nai):
                if not np.isfinite(aoI[i]):
                    f.write('{}{}\n'.format(space, substituteI))
                else:
                    f.write('{}{}\n'.format(space, aoI[i]))
    else:
        if atype == "Float64":
            atD = np.array(array, dtype='double')
            for i in range(zeros):
                f.write(space)
                for j in range(Naj[0]):
                    f.write('{} '.format(zeroD))
                f.write('\n')
            for i in range(Nai):
                f.write(space)
                for j in range(Naj[i]):
                    if not np.isfinite(atD[i, j]):
                        f.write('{} '.format(substituteD))
                    else:
                        f.write('{} '.format(atD[i, j]))
                f.write('\n')
        elif atype == "Int64":
            atI = np.array(array, dtype=np.int32)
            for i in range(zeros):
                f.write(space)
                for j in range(Naj[0]):
                    f.write('{} '.format(zeroI))
                f.write('\n')
            for i in range(Nai):
                f.write(space)
                for j in range(Naj[i]):
                    if not np.isfinite(atI[i, j]):
                        f.write('{} '.format(substituteI))
                    else:
                        f.write('{}'.format(atI[i, j]))
                f.write('\n')
    f.write('{}</DataArray>\n'.format(4 * tab))


def export_microbloom_setup_files(graph, diameter_attribute = 'diameter',p_in_cow_mmHg=100., p_out_av_mmHg=10.,
                               write_inverse_model_data=False,
                               target_eids=np.array([]), target_value_min=np.array([]), target_value_max=np.array([]),
                               sigma=np.array([]), value_type=np.array([]),
                               parameter_eids = np.array([]), parameter_Delta = np.array([]), path=""):


    ################
    # Write network
    ################

    diameter = np.array(graph.es[diameter_attribute]) * 1e-6 # convert to m
    adjacency_list = np.array(graph.get_edgelist(), dtype=int)
    length = np.array(graph.es['length']) * 1e-6  # convert to m

    # vertex data
    vertex_positions = np.array(graph.vs['coords']) * 1e-6  # convert to m
    x = vertex_positions[:, 0]
    y = vertex_positions[:, 1]
    z = vertex_positions[:, 2]

    # boundary data at outlet (AVs) and inlet (cow)
    p_out_av = p_out_av_mmHg * 133.322  # converted to Pa
    p_in_cow = p_in_cow_mmHg * 133.322  # converted to Pa

    v_ids_all = np.arange(graph.vcount())

    is_inflow = np.array(graph.vs['COW_in']) > 0
    inflow_vs = v_ids_all[is_inflow]
    inflow_type = np.ones(np.size(inflow_vs)) * 1.  # 1: pressure, 2: flux
    inflow_press = np.ones(np.size(inflow_vs)) * p_in_cow
    inflow_flux = np.ones(np.size(inflow_vs))
    inflow_flux[:] = np.nan

    is_outflow = np.array(graph.vs['is_AV_root']) > 0
    outflow_vs = v_ids_all[is_outflow]
    outflow_type = np.ones(np.size(outflow_vs)) * 1.  #: pressure, 2: flux
    outflow_press = np.ones(np.size(outflow_vs)) * p_out_av
    outflow_flux = np.ones(np.size(outflow_vs))
    outflow_flux[:] = np.nan

    bc_vs = np.append(inflow_vs, outflow_vs)
    bc_type = np.append(inflow_type, outflow_type)
    bc_press = np.append(inflow_press, outflow_press)
    bc_flux = np.append(inflow_flux, outflow_flux)

    # write network data (vertex, edge and boundary data)
    export_node_data(x, y, z, filepath=path+"node_data.csv")
    export_edge_data(adjacency_list, diameter, length, filepath=path+"edge_data.csv")
    export_boundary_node_data_dirichletAndNeuman(bc_vs, bc_type, bc_press, bc_flux, filepath=path+"boundary_node_data.csv")

    ########################
    # Write inverse problem
    # input files
    ########################

    if write_inverse_model_data:

        export_matrix_param = np.zeros((np.size(parameter_eids),2))
        export_matrix_param[:,0] = parameter_eids
        export_matrix_param[:,1] = parameter_Delta
        header = [['edge_param_eid','edge_param_pm_range']]
        with open(path+"parameters_complete_data.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(header)
            writer.writerows(export_matrix_param)

        # Write target for edges
        target_value_mean = .5 * (target_value_max + target_value_min)
        target_value_range = .5 * (target_value_max - target_value_min)

        export_matrix = np.zeros((np.size(target_eids), 5))
        export_matrix[:, 0] = target_eids
        export_matrix[:, 1] = value_type
        export_matrix[:, 2] = target_value_mean
        export_matrix[:, 3] = target_value_range
        export_matrix[:, 4] = sigma     # 0: flux, 1: urbc

        header = [['edge_tar_eid', 'edge_tar_type', 'edge_tar_value', 'edge_tar_range_pm', 'edge_tar_sigma']]
        with open(path+"edge_target_data.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(header)
            writer.writerows(export_matrix)

    return
