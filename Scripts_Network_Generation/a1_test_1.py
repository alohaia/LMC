#!/usr/bin/env python

import datetime
import pickle
import sys
import time

import igraph
from igraph import layout
from igraph.drawing import graph
import numpy as np
import pandas as pd

from source.export_graph import write_vtp
from source.graph_manipulation import (
    add_AV_starting_pts_to_graph,
    connect_all_new_DAs_starting_pt_to_graph,
)
from source.graph_manipulation_hardcoded import (
    add_connection2_cow_balbc_1,
    add_connection2_cow_balbc_2,
    add_connection2_cow_c57bl6_1,
    add_connection2_cow_c57bl6_2,
    add_connection2_cow_test_1,
    add_connection2_cow_test_1,
    change_graph_attributes_casespecific_balbc_1,
    change_graph_attributes_casespecific_balbc_2,
    change_graph_attributes_casespecific_c57bl6_1,
    change_graph_attributes_casespecific_c57bl6_2,
    change_graph_attributes_casespecific_test_1,
)
from source.import_graph import (
    check_graph_attributes_complete_nwgen,
    import_sa_graph_from_pkl,
)
from source.network_visz import (
    plot_pial_vasculature,
    visualize_DA_AV_roots_with_polygons,
)
from source.processing_spatial import (
    add_AVs,
    do_DA_density_refinement,
    write_AV_locs_to_file,
    write_voronoi_polygons_DAs_to_file,
)

##################################################################
# Import a surface artery network from two python dictionaries,
# which contain the topology of the incomplete surface artery
# networks.
#################################################################

# Network C57BL/6_I
# graph_surface_artery = import_sa_graph_from_pkl('surface_artery_networks/C57BL6_1/edgesDict.pkl',
#                                                 'surface_artery_networks/C57BL6_1/verticesDict.pkl',
#                                                 adj_attr_name='connectivity', delete_tortiosity=True)

vs_data = pd.read_csv("../data/test 1/vertices.csv")
es_data = pd.read_csv("../data/test 1/edges.csv")

g = igraph.Graph((es_data[["from", "to"]] - 1).to_numpy().tolist())

ts = time.time()
g['network_gen_date'] = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
g['Mouse'] = "Test 1"

vs_coords = vs_data.loc[:, ["X", "Y"]].to_numpy()
vs_coords = np.hstack([vs_coords, np.zeros((vs_coords.shape[0], 1))])
g.vs["coords"] = vs_coords
g.vs["MCA_in"] = vs_data.MCA_in
g.vs["ACA_in"] = vs_data.ACA_in
g.vs["is_DA_startingPt"] = vs_data.is_DA_startingPt # FIXME
g.vs["is_DA_startingPt_added_manually"] = np.zeros(g.vcount(), dtype=int)
g.vs['is_AV_root'] = np.zeros(g.vcount(), dtype=int)
g.vs['is_connected_2caps'] = np.zeros(g.vcount(), dtype=int)
g.vs['COW_in'] = np.zeros(g.vcount(),dtype=int)


g.es['is_stroke'] = es_data.is_stroke
g.es["diameter"] = es_data.diameter
g.es["isCollateral"] = es_data.isCollateral
g.es['type'] = es_data.type
g.es["added_manually"] = es_data.added_manually

# FIXME
g.es["index_exp"] = es_data.index_exp
g.es["diam_pre_exp"] = es_data.diam_pre_exp
g.es["diam_post_exp"] = es_data.diam_post_exp
g.es["vRBC_pre_exp"] = es_data.vRBC_pre_exp
g.es["vRBC_post_exp"] = es_data.vRBC_post_exp


lengths = []
for e in g.es:
    v1, v2 = e.tuple
    p1, p2 = g.vs[v1]["coords"], g.vs[v2]["coords"]
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    lengths.append(dist)
g.es["length"] = lengths


igraph.plot(
    g,
    layout=[arr[:2] for arr in g.vs["coords"]],
    vertex_size=4,
    vertex_color="lightblue",
    edge_color="gray",
    bbox=(600, 600),
    margin=30
)

graph_surface_artery = g


##################################################################
# Network specific modifications, i.e., to make sure that all graph
# attributes are consistent across the datasets. This required
# to manually change, add and remove some attributes.
# Note that a different method is used for every network
##################################################################

# Network C57BL/6_I
graph_surface_artery = change_graph_attributes_casespecific_test_1(graph_surface_artery)

# Network C57BL/6_II
# graph_surface_artery = change_graph_attributes_casespecific_c57bl6_2(graph_surface_artery)

# Network BALB/C_I
# graph_surface_artery = change_graph_attributes_casespecific_balbc_1(graph_surface_artery)

# Network BALB/C_II
# graph_surface_artery = change_graph_attributes_casespecific_balbc_2(graph_surface_artery)

##################################################################
# Check if the network contains all necessary attributes and
# print a summary of the network attributes to the terminal
##################################################################

if not check_graph_attributes_complete_nwgen(graph_surface_artery):
    sys.exit()

print('Graph summary after SA import')
print(graph_surface_artery.summary())

##################################################################
# Add additional surface arteries (SA to CoW) to connect MCA_in
# and ACA_in to a common inflow vertex (COW_in).
# This is network specific and done manually (hardcoded).
##################################################################

# Network C57BL/6_I
graph_surface_artery = add_connection2_cow_test_1(graph_surface_artery)

# Network C57BL/6_II
# graph_surface_artery = add_connection2_cow_c57bl6_2(graph_surface_artery)

# Network BALB/C_I
# graph_surface_artery = add_connection2_cow_balbc_1(graph_surface_artery)

# Network BALB/C_II
# graph_surface_artery = add_connection2_cow_balbc_2(graph_surface_artery)

##################################################################
# Refine the density of DA trees, i.e., the DA roots (C57BL/6: 13.4 DAs/mm2; BALB/C: 8.9 DAs/mm2)
##################################################################

# The x/y coordinates of all experimentally detected DA roots are used as input for the density refinement of DA trees.
x = np.array(graph_surface_artery.vs["coords"])[:, 0]  # x coordinates of all vertices
y = np.array(graph_surface_artery.vs["coords"])[:, 1]  # y coordinates of all vertices
is_DA_startingPt = np.array(graph_surface_artery.vs["is_DA_startingPt"]) > 0  # identify if vertex is a DA root
is_ACA_in = np.array(graph_surface_artery.vs["ACA_in"]) > 0  # identify if vertex is ACA inflow
is_MCA_in = np.array(graph_surface_artery.vs["MCA_in"]) > 0  # identify if vertex is MCA inflow

# 2d numpy array with x and y coordinates of all DA root vertices
xy_DA_roots = np.array([x[is_DA_startingPt], y[is_DA_startingPt]]).transpose()
# 2d numpy array with x and y coordinates of all MCA and ACA inflows
xy_ACA_MCA = np.array([x[np.logical_or(is_MCA_in,is_ACA_in)], y[np.logical_or(is_MCA_in,is_ACA_in)]]).transpose()
# 2d numpy array with x/y coordinates of DA roots and MCA/ACA inflows: Used as input for refinement algorithm
xy_coords_input = np.concatenate((xy_DA_roots, xy_ACA_MCA),axis=0)

# Calls the refinement algorithm described in S1 Appendix of the manuscript. Note that the parameter
# newDA_prior_target_density corresponds to 1/A_lim and indirectly defines the final average DA density (can be checked
# after refinement in the file nw_output/DA_AV_locations/density_distr_refinement_nr_xxx.png).
# Values for newDA_prior_target_density of all networks (note that this parameter varies slightly for different networks
# and was chosen as large as possible, while ensuring that the prescribed average target density was matched):
# C57BL/6_I: 11.5; C57BL/6_II: 11.8; BALB/C_I: 6.65; BALB/C_II: 7.1
xy_DA_roots_new_points, polygon_vs_xy = do_DA_density_refinement(xy_DA_roots, max_nr_of_new_DAs=200,
                                                                 ghp_mode=2, ghp_frame_width=800, ghp_hull_meshsize=100,
                                                                 ghp_shape_simplifier=10, sample_min_DA_distance=200.,
                                                                 newDA_prior_target_density=11.5, max_nr_of_tries=50000,
                                                                 write_refinement_steps_to_file=False,
                                                                 write_init_final_distribution=True,
                                                                 filepath_folder="nw_output/DA_AV_locations/",
                                                                 return_voronoi_polygon_vs_coords=True)

# Writes the coordinates of all DA roots and the coordinates of the voronoi Polygons to file
write_voronoi_polygons_DAs_to_file(polygon_vs_xy, xy_DA_roots, xy_DA_roots_new_points, filepath="nw_output/DA_AV_locations/")

##################################################################
# Add the locations of ascending vein (AV) trees (AV roots) on the Voroni polygons and visualise the locations
##################################################################

xy_DA_all = np.concatenate((xy_DA_roots, xy_DA_roots_new_points), axis=0)
xy_AVs = add_AVs(polygon_vs_xy, xy_DA_all, 3, min_dist_2_DA = 100, min_dist_2_AV = 120, max_nr_of_tries=200000)

title_plot = "Nr of DAs: " + str(np.size(xy_DA_all, 0)) + ", Nr of AVs: " + str(np.size(xy_AVs, 0))
is_initial_DA = np.array([True]*np.size(xy_DA_all,0))
is_initial_DA[np.size(xy_DA_roots,0):] = False
visualize_DA_AV_roots_with_polygons(xy_DA_all, is_initial_DA, xy_AVs, polygon_vs_xy, show_MCA_ACA_root=False,
                                    title=title_plot, filepath="nw_output/DA_AV_locations/DA_AV_map.png")

write_AV_locs_to_file(xy_AVs, filepath="nw_output/DA_AV_locations/")

##################################################################
# Connect the newly sampled DA root points to the surface artery network
# and visualize the resulting surface artery network
##################################################################

graph_surface_artery, is_valid = connect_all_new_DAs_starting_pt_to_graph(graph_surface_artery, xy_DA_roots_new_points,
                                                                          target_new_min_splitedgelength=75,
                                                                          max_distance_new_DA=1000, distort=True,
                                                                          distort_max=.3)

# Network checks
if not is_valid:
    print("Error: Could not connect all DAs due to error")
    sys.exit()
if not check_graph_attributes_complete_nwgen(graph_surface_artery):
    sys.exit()

# Visualization
x = np.array(graph_surface_artery.vs["coords"])[:, 0]
y = np.array(graph_surface_artery.vs["coords"])[:, 1]
adjacency_list = np.array(graph_surface_artery.get_edgelist(), dtype=int)
diameters = np.array(graph_surface_artery.es["diameter"])
is_collateral = np.array(graph_surface_artery.es["is_collateral"]) > 0
is_added_manually = np.array(graph_surface_artery.es["added_manually"]) > 0

is_DA_startingPt = np.array(graph_surface_artery.vs["is_DA_startingPt"]) > 0
is_DA_startingPt_added_manually = np.array(graph_surface_artery.vs["is_DA_startingPt_added_manually"]) > 0
is_MCA_in = np.array(graph_surface_artery.vs["MCA_in"]) > 0
is_ACA_in = np.array(graph_surface_artery.vs["ACA_in"]) > 0

plot_pial_vasculature(x, y, adjacency_list, diameters, is_collateral, is_added_manually, is_DA_startingPt,
                      is_DA_startingPt_added_manually, is_MCA_in, is_ACA_in, xy_AV_roots=xy_AVs, show_vs_ids=True,
                      title="B6_B (1st set, B6_B)", filepath="nw_output/DA_AV_locations/graph_refined_nw.png")

##################################################################
# Add the newly sampled AV root points as vertices to the network
##################################################################

graph_surface_artery = add_AV_starting_pts_to_graph(graph_surface_artery, xy_AVs)

##################################################################
# Check network for the consistency of all network attributes
##################################################################
if not check_graph_attributes_complete_nwgen(graph_surface_artery):
    sys.exit()

##################################################################
# Write the refined surface artery network to two files:
# - igraph pkl file, will be further used for the next steps in the network generation process
# - vtp file, can be opened with paraview and used for the visualization of the network
##################################################################

igraph.plot(graph_surface_artery, layout=[arr[:2] for arr in graph_surface_artery.vs["coords"]])

graph_surface_artery.write_pickle('nw_output/graph_gen_process/a1_test_1_surface_network.pkl')
write_vtp(graph_surface_artery, 'nw_output/graph_gen_process/a1_test_1_surface_network.vtp', coordinatesKey='coords')
