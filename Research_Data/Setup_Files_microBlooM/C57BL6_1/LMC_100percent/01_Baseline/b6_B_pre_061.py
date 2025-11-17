"""
A python script to simulate stationary blood flow in microvascular networks with considering the vessel distensibility.
Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Save the results in a file
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
        "csv_path_vertex_data": "testcases/paper_lmc_plos/b6_B/b6_B_pre_061/data/node_data.csv",
        "csv_path_edge_data": "testcases/paper_lmc_plos/b6_B/b6_B_pre_061/data/edge_data.csv",
        "csv_path_boundary_data": "testcases/paper_lmc_plos/b6_B/b6_B_pre_061/data/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Write options
        "write_path_igraph": "testcases/paper_lmc_plos/b6_B/b6_B_pre_061/output/b6_B_pre_061_results",  # only required for "write_network_option" 2, 3, 4
    }
)

# Create object to set up the simulation and initialise the simulation
setup_blood_flow = setup.SetupSimulation()
# Initialise the implementations based on the parameters specified
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_blood_flow.setup_bloodflow_model(PARAMETERS)

# Build flownetwork object and pass the implementations of the different submodules, which were selected in
#  the parameter file
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)

# Import or generate the network
print("Read network: ...")
flow_network.read_network()
print("Read network: DONE")

# Flow solver
print("Solve flow: ...")
flow_network.update_transmissibility()
flow_network.update_blood_flow()
print("Solve flow: DONE")

flow_network.write_network()
