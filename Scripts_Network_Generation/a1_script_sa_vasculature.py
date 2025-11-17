import igraph
import sys

from utils.import_graph import check_graph_attributes_complete_nwgen
from utils.graph_manipulation import add_capillary_bed, remove_regions_outside_DAs
from utils.export_graph import write_vtp

##################################################################
# Import the SA/DA/AV network, which was generated in the previous step with the script
# a2_script_add_penetrating_trees.py
##################################################################

graph_sa_da_av = igraph.Graph.Read_Pickle('nw_output/graph_gen_process/a2_c57bl6_1_sa_da_av_network.pkl')

# Print the network summary and check the attributes
print(graph_sa_da_av.summary())
if not check_graph_attributes_complete_nwgen(graph_sa_da_av):
    sys.exit()

##################################################################
# Generate and add the artificial capillary bed (this can take a long time, since the functions to connect DA and AV
# leaf vertices to capillaries are very slow).
# d_vessel and l_vessel define vessel diameter and tortuous length.
##################################################################

graph_sa_da_av_c = add_capillary_bed(graph_sa_da_av, z_min_caps=25., frame_bounding_box=50.,
                                     distance_between_cap_vs=45., l_vessel=62., d_vessel=4.,
                                     perturb_vs_frac=.8)

##################################################################
# Check consistency of the network attributes
##################################################################

if not check_graph_attributes_complete_nwgen(graph_sa_da_av_c):
    sys.exit()

##################################################################
# Clean-up the network by removing vessels that are located in regions away from the original surface artery network
##################################################################

graph_final = remove_regions_outside_DAs(graph_sa_da_av_c, 500.)

print(graph_final.summary())
print("Graph is connected:", graph_final.is_connected())
# If the graph is not connected: manually remove the smaller components. Show the components:
# print g_b6_b_cropped.components()

if not check_graph_attributes_complete_nwgen(graph_final):
    sys.exit()

##################################################################
# Write the final semi-realistic network to two files:
# - igraph pkl file, will be further used for the next steps in the network generation process
# - vtp file, can be opened with paraview and used for the visualization of the network
##################################################################

graph_final.write_pickle('nw_output/graph_gen_process/semi_realistic_network_final.pkl')
write_vtp(graph_final, 'nw_output/graph_gen_process/semi_realistic_network_final.vtp', coordinatesKey='coords')
