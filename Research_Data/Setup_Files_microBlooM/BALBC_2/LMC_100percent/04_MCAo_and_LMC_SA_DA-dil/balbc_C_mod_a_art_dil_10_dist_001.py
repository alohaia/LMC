"""
A python script to simulate stationary blood flow in microvascular networks with considering the vessel distensibility.
Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Update the vessel diameters based on the current pressure distribution
5. Save the results in a file
"""
import sys
import numpy as np

from source.flow_network import FlowNetwork
from source.distensibility import Distensibility
from types import MappingProxyType
import source.setup.setup as setup


# MappingProxyType is basically a const dict.
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 2,  # 1: generate hexagonal graph
                                   # 2: import graph from csv files
                                   # 3: import graph from igraph file (pickle file)
        "write_network_option": 4,  # 1: do not write anything
                                    # 3: write to vtp format
                                    # 4: write to two csv files
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
        "rbc_impact_option": 2,  # 1: No RBCs (hd=0)
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: Laws by Pries and Secomb (2005)
        "solver_option": 1,  # 1: Direct solver
                             # 2: PyAMG solver

        # Blood properties
        "ht_constant": 0.3,  # only required if RBC impact is considered
        "mu_plasma": 0.0012,

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "testcases/paper_lmc_plos/balbc_C_mod_a/balbc_C_mod_a_art_dil_10_dist_001/data/node_data.csv",
        "csv_path_edge_data": "testcases/paper_lmc_plos/balbc_C_mod_a/balbc_C_mod_a_art_dil_10_dist_001/data/edge_data.csv",
        "csv_path_boundary_data": "testcases/paper_lmc_plos/balbc_C_mod_a/balbc_C_mod_a_art_dil_10_dist_001/data/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Write options
        "write_path_igraph": "testcases/paper_lmc_plos/balbc_C_mod_a/balbc_C_mod_a_art_dil_10_dist_001/output/balbc_C_mod_a_art_dil_10_dist_001_results",  # only required for "write_network_option" 2, 3, 4

        ##########################
        # Vessel distensibility options
        ##########################

        # Set up distensibility model
        "distensibility_model": 3,   # 1: No update of diameters due to vessel distensibility
                                     # 2: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = p_base, d_ref = d_base
                                     # 3: Passive diam changes, tube law. 1/D_ref ≈ 1/D. p_ext = const, d_ref computed.

        # Distensibility edge properties
        "csv_path_distensibility": "testcases/paper_lmc_plos/balbc_C_mod_a/balbc_C_mod_a_art_dil_10_dist_001/data/balbc_C_mod_a_art_dil_10_dist_001_properties.csv",
        "pressure_external": 0.  # Constant external pressure as reference pressure (only for distensibility_model 2)
    }
)

# Create object to set up the simulation and initialise the simulation
setup_blood_flow = setup.SetupSimulation()
# Initialise the implementations based on the parameters specified
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_blood_flow.setup_bloodflow_model(PARAMETERS)

imp_distensibility_law, imp_read_distensibility_parameters = setup_blood_flow.setup_distensibility_model(PARAMETERS)

# Build flownetwork object and pass the implementations of the different submodules, which were selected in
#  the parameter file
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)

distensibility = Distensibility(flow_network, imp_distensibility_law, imp_read_distensibility_parameters)

# Import or generate the network
print("Read network: ...")
flow_network.read_network()
print("Read network: DONE")

# Diameters after stroke, without distensibility
diameter_stroke_init = np.copy(flow_network.diameter)

# Diameters at baseline. Are needed to compute the reference pressure and diameters
import pandas as pd
diameter_baseline = pd.read_csv("testcases/paper_lmc_plos/balbc_C_mod_a/balbc_C_mod_a_art_dil_10_dist_001/data/balbc_C_mod_a_diameters_pre_stroke.csv")["d_base"].to_numpy()
flow_network.diameter = diameter_baseline

# Baseline
print("Solve baseline flow (for reference): ...")
flow_network.update_transmissibility()
flow_network.update_blood_flow()
print("Solve baseline flow (for reference): DONE")

# Initialise distensibility model based on baseline (pre-stroke) diameters and pressures
print("Initialise distensibility model based on baseline results: ...")
distensibility.initialise_distensibility()
print("Initialise distensibility model based on baseline results: DONE")

# Add stroke
flow_network.diameter = diameter_stroke_init

# Update diameters and iterate (has to be improved)
tol = 1.e-10
diameters_current = flow_network.diameter  # Previous diameters to monitor convergence of diameters
for i in range(10):
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    distensibility.update_vessel_diameters()
    print("Distensibility update: it=" + str(i + 1) + ", residual = " + "{:.2e}".format(
        np.max(np.abs(flow_network.diameter - diameters_current))) + " um (tol = " + "{:.2e}".format(tol)+")")
    if np.max(np.abs(flow_network.diameter - diameters_current)) < tol:
        print("Distensibility update: DONE")
        break
    else:
        diameters_current = flow_network.diameter

flow_network.write_network()
