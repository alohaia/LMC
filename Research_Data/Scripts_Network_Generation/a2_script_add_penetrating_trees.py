import igraph
import numpy as np
import sys

from source.import_graph import check_graph_attributes_complete_nwgen
from source.graph_manipulation import add_da_trees_to_sa_graph, add_av_trees_to_sa_graph
from source.export_graph import write_vtp

##################################################################
# Import the refined surface artery network, which was generated in the previous step with the script
# a1_script_sa_vasculature.py
##################################################################

graph_surface_artery = igraph.Graph.Read_Pickle('nw_output/graph_gen_process/a1_c57bl6_1_surface_network.pkl')

# Print the network summary and check the attributes
print graph_surface_artery.summary()
if not check_graph_attributes_complete_nwgen(graph_surface_artery):
    sys.exit()

##################################################################
# Add DA trees to the network by sampling from a database
##################################################################

da_candidates = np.array([0, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                          21, 22, 23, 24, 26, 27, 28, 29,
                          32, 33, 35, 36, 37, 39, 42, 44, 45, 46, 47, 48, 49,
                          50, 52, 54, 55, 56], dtype=np.int)

graph_sa_da = add_da_trees_to_sa_graph(graph_surface_artery, da_candidates, 'penetrating_trees/arteries/')

print 'Graph summary after merging SAs with DAs',graph_sa_da.summary()

##################################################################
# Add AV trees to the network by sampling from a database
##################################################################

av_candidates = np.array(
    [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 39, 42, 43, 44,
     45, 47, 50, 51, 52, 54, 57, 58, 59, 66, 67, 69, 71, 73, 74, 75, 76, 79, 80, 81, 82, 86, 88, 90, 91, 92, 93, 97, 98,
     99],
    dtype=np.int)

graph_sa_da_av = add_av_trees_to_sa_graph(graph_sa_da, av_candidates, 'penetrating_trees/veins/')

print 'Graph summary after merging SAs with DAs & AVs',graph_sa_da_av.summary()

##################################################################
# Check consistency of the network attributes
##################################################################
if not check_graph_attributes_complete_nwgen(graph_sa_da_av):
    sys.exit()

##################################################################
# Write the network which now includes surface arteries and penetrating trees to two files:
# - igraph pkl file, will be further used for the next steps in the network generation process
# - vtp file, can be opened with paraview and used for the visualization of the network
##################################################################

graph_sa_da_av.write_pickle('nw_output/graph_gen_process/a2_c57bl6_1_sa_da_av_network.pkl')
write_vtp(graph_sa_da_av, 'nw_output/graph_gen_process/a2_c57bl6_1_sa_da_av_network.vtp', coordinatesKey='coords')
