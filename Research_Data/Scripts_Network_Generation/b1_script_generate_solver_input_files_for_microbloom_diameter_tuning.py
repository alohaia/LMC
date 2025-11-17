import igraph
import numpy as np
import sys

from source.import_graph import check_graph_attributes_complete_nwgen
from source.network_analysis import get_eids_vs_roots
from source.export_graph import export_microbloom_setup_files

##################################################################
# Import the semi-realistic microvascular network, which was generated with the script
# a3_script_add_capillaries.py
##################################################################

graph_before_tuning = igraph.Graph.Read_Pickle('nw_output/graph_gen_process/semi_realistic_network_final.pkl')

# Print the network summary and check the attributes
print graph_before_tuning.summary()
if not check_graph_attributes_complete_nwgen(graph_before_tuning):
    sys.exit()

#######################################################################
# Identify the edge ids with velocity and diameter measurements.
# Note that this step is dataset specific.
#######################################################################

# ID of vessels with a velocity measurement (note that these numbers have been renumbered in S1 Table of the manuscript)
exp_urbc_meas_ind_usz = np.array([12, 6, 21, 22, 23, 13, 14, 15, 16, 17, 19, 20], dtype=np.int)
is_urbc_meas_usz = np.in1d(np.array(graph_before_tuning.es['index_exp']), exp_urbc_meas_ind_usz)
eids_urbc_meas_usz = np.arange(graph_before_tuning.ecount())[is_urbc_meas_usz]  # edge ids of measurement locations
vals_urbc_meas_usz = np.array(graph_before_tuning.es['vRBC_pre_exp'])[is_urbc_meas_usz]  # uRBC values of measurements

# ID of vessels with a diameter measurement (note that these numbers have been renumbered in S1 Table of the manuscript)
exp_diam_meas_ind_usz = np.array([1, 9, 10, 11, 12, 6, 5, 21, 22, 23, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.int)
is_diam_meas_usz = np.in1d(np.array(graph_before_tuning.es['index_exp']), exp_diam_meas_ind_usz)

#######################################################################
# Set additional value in LMCs without measurements (median of other LMCs)
#######################################################################

eids_manual = np.array([6])  # Here, only one additional target value in collateral
uRBCs_manual = np.array([-.53])  # Median value, mm/s. sign chosen such that flow MCA->ACA

#######################################################################
# Edge ids and velocity target values for setting up the inverse problem in microBloom
#######################################################################

e_ids_target_measurement = np.append(eids_urbc_meas_usz, eids_manual)
uRBC_target_meas = np.append(vals_urbc_meas_usz, uRBCs_manual) / 1000.   # Convert velocity to meter per second

for eid, urbc in zip(e_ids_target_measurement, uRBC_target_meas):
    print eid, graph_before_tuning.get_edgelist()[eid], urbc

vRBC_target = np.array([None] * graph_before_tuning.ecount())
vRBC_target[e_ids_target_measurement] = uRBC_target_meas
graph_before_tuning.es['vRBC_target_pre_si'] = vRBC_target  # Assign the target velocities to an edge attribute

is_target_edge = np.zeros(graph_before_tuning.ecount(), dtype=np.int)
is_target_edge[e_ids_target_measurement] = 1
graph_before_tuning.es['is_target_edge'] = is_target_edge  # Edge attribute which is True if current edge has a measurement

# Find the vertex and edge indices of DA and AV root
vids_roots_DA, eids_roots_DA = get_eids_vs_roots(graph_before_tuning, type=2)
vids_roots_AV, eids_roots_AV = get_eids_vs_roots(graph_before_tuning, type=3)

urbc_min_target_root_DA = np.ones(np.size(eids_roots_DA)) * .002  # Minimum velocity in DAs
urbc_max_target_root_DA = np.ones(np.size(eids_roots_DA)) * .010  # Maximum velocity in DAs

urbc_min_target_root_AV = np.ones(np.size(eids_roots_AV)) * (-.002)  # Minimum velocity in AVs
urbc_max_target_root_AV = np.ones(np.size(eids_roots_AV)) * (-.0004)  # Maximum velocity in AVs

eids_roots = np.append(eids_roots_DA, eids_roots_AV)  # Edge ids of all DA and AV roots
urbc_min_target_root = np.append(urbc_min_target_root_DA, urbc_min_target_root_AV)
urbc_max_target_root = np.append(urbc_max_target_root_DA, urbc_max_target_root_AV)

target_eids = np.append(e_ids_target_measurement, eids_roots)  # All edge ids with a target value or range
target_value_min = np.append(uRBC_target_meas, urbc_min_target_root)  # Target velocities minimum values
target_value_max = np.append(uRBC_target_meas, urbc_max_target_root)  # Max values
sigma = np.ones(np.size(target_eids)) * 0.025
value_type = 2*np.ones(np.size(target_eids))  # 2: velocity

###############################################
# Edge ids and maximum diameter change for setting up the inverse problem in microBloom
###############################################

vessel_type_all = np.array(graph_before_tuning.es['type'])

parameter_eids = np.arange(graph_before_tuning.ecount(), dtype=np.int)

parameter_Delta = np.ones(np.size(parameter_eids)) * 0.5  # generally 50%
parameter_Delta[vessel_type_all<=0] = 0.2  # SAs and SA2CoW 20%

parameter_Delta[is_diam_meas_usz] = 0.05  # allow 5 % for SAs with diameter measurement
parameter_Delta[graph_before_tuning.get_eid(12,91)] = 0.05  # Stroke edge (network specific)
parameter_Delta[graph_before_tuning.get_eid(89,91)] = 0.05  # Stroke edge, occlusion part (network specific)
parameter_Delta[graph_before_tuning.get_eid(90,89)] = 0.05  # Upstream to stroke bifurcation (network specific)
parameter_Delta[graph_before_tuning.get_eid(88,90)] = 0.05  # At COW MCA (network specific)
parameter_Delta[graph_before_tuning.get_eid(88,87)] = 0.05  # At COW ACA (network specific)

# Save parameters to graph
is_parameter_inverse_prob = np.zeros(graph_before_tuning.ecount())
is_parameter_inverse_prob[parameter_eids] = parameter_Delta
graph_before_tuning.es['is_parameter_inverse_prob'] = is_parameter_inverse_prob   # Assign the parameter as edge attribute

###############################################
# Write the setup files for tuning the networks with microbloom
###############################################

export_microbloom_setup_files(graph_before_tuning, p_in_cow_mmHg=100., p_out_av_mmHg=10., write_inverse_model_data=True,
                           target_eids=target_eids, target_value_min=target_value_min, target_value_max=target_value_max,
                           sigma=sigma, value_type=value_type, parameter_eids = parameter_eids,
                           parameter_Delta = parameter_Delta, path="nw_output/setup_microbloom_diameter_tuning/")


################
# Write to file
################

graph_before_tuning.write_pickle('nw_output/setup_microbloom_diameter_tuning/network_before_tuning.pkl')
