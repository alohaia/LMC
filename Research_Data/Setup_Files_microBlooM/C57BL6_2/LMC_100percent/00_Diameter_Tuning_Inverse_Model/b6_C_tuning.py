"""
A python script to estimate edge parameters such as diameters and transmissibilities of microvascular networks based
 on given flow rates and velocities in selected edges.
 Run the script from main.py
 Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Update the diameters and transmissibilities with a gradient descent algorithm minimising a given cost function.
5. Restriction of parameters to desired ranges (target value +/- tolerance).
6. Individual selection of parameter edges and target edges.
7. Target flow rates and velocities can be specified and combined into a single cost function.
8. Tuning of either relative diameters or relative transmissibilities compared to baseline.
9. Optimisation of diameters for a fixed number of iteration steps.
10. Save the results in a file.
"""

from source.flow_network import FlowNetwork
from source.inverse_model import InverseModel
from types import MappingProxyType
import source.setup.setup as setup

# MappingProxyType is basically a const dict
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 2,  # 2: import graph from csv files
        "write_network_option": 4,  # 4: write to two csv files
        "tube_haematocrit_option": 2,  # 2: Constant haematocrit
        "rbc_impact_option": 2,  # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
        "solver_option": 1,  # 1: Direct solver

        # Blood properties
        "ht_constant": 0.3,  # Constant tube haematocrit in all vessels
        "mu_plasma": 0.0012,

        # Import network from csv options - Only required for "read_network_option": 2
        "csv_path_vertex_data": "testcases/paper_lmc_plos/b6_C/b6_C_tuning_setup/data/node_data.csv",
        "csv_path_edge_data": "testcases/paper_lmc_plos/b6_C/b6_C_tuning_setup/data/edge_data.csv",
        "csv_path_boundary_data": "testcases/paper_lmc_plos/b6_C/b6_C_tuning_setup/data/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Write options
        "write_path_igraph": "testcases/paper_lmc_plos/b6_C/b6_C_tuning_setup/output/b6_C_tuning_results",

        ##########################
        # Inverse problem options
        ##########################

        # Define parameter space
        "parameter_space": 1,  # 1: Relative diameter to prior diameter distribution (alpha = d/d_base)
        "parameter_restriction": 2,  # 2: Restriction of parameter by a +/- tolerance to baseline
        "inverse_model_solver": 1,  # 1: Direct solver

        # Target edges
        "csv_path_edge_target_data": "testcases/paper_lmc_plos/b6_C/b6_C_tuning_setup/data/edge_target_data.csv",
        # Parameter edges
        "csv_path_edge_parameterspace": "testcases/paper_lmc_plos/b6_C/b6_C_tuning_setup/data/parameters_complete_data.csv",
        # Gradient descent options:
        "gamma": 5,
        "phi": .5,
        "max_nr_of_iterations": 100000,
        "convergence_criteria": 1.e-4
    }
)

setup_simulation = setup.SetupSimulation()

# Initialise objects related to simulate blood flow without RBC tracking (constant haematocrit).
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_simulation.setup_bloodflow_model(PARAMETERS)

# Initialise objects related to the inverse model.
imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter, imp_adjoint_solver, \
    imp_alpha_mapping = setup_simulation.setup_inverse_model(PARAMETERS)

# Initialise flownetwork and inverse model objects
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)
inverse_model = InverseModel(flow_network, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                             imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

print("Read network: ...")
flow_network.read_network()  # Import the network from csv files
print("Read network: DONE")

print("Update transmissibility: ...")
flow_network.update_transmissibility()  # Update the edge transmissibilities
print("Update transmissibility: DONE")

print("Update flow, pressure and velocity: ...")
flow_network.update_blood_flow()  # Compute flow rates, pressures and RBC velocities
print("Update flow, pressure and velocity: DONE")

inverse_model.initialise_inverse_model()  # Initialise the inverse model
inverse_model.update_cost()

nr_of_iterations = int(PARAMETERS["max_nr_of_iterations"])
convergence_criteria = PARAMETERS["convergence_criteria"]

print("Solve the inverse problem and update the diameters: ...")

it = 0
print(str(it) + " / " + str(nr_of_iterations) + " iterations done (f_H =", "%.2e" % inverse_model.f_h + ")")

# Run the inverse model and iterate
while (it < nr_of_iterations) and (inverse_model.f_h > convergence_criteria):
    inverse_model.update_state()  # Update the diameters to match the prescribed velocities
    flow_network.update_transmissibility()
    flow_network.update_blood_flow()
    inverse_model.update_cost()  # Update the new cost function values
    it += 1

    if it % 10 == 0:
        print(str(it)+" / " + str(nr_of_iterations) + " iterations done (f_H =", "%.2e" % inverse_model.f_h+")")

print(str(nr_of_iterations)+" / " + str(nr_of_iterations) + " iterations done (f_H =", "%.2e" % inverse_model.f_h+")")
print("Solve the inverse problem and update the diameters: DONE")

flow_network.write_network()  # write the results to files

print("Type\t\tEid\t\tVal_tar_min\t\tVal_tar_max,\tVal_opt,\tVal_base ")
for eid, value, range, type in zip(inverse_model.edge_constraint_eid, inverse_model.edge_constraint_value, inverse_model.edge_constraint_range_pm, inverse_model.edge_constraint_type):
    if type==1:
        print("Flow rate","\t",eid,"\t",value-range,"\t","\t",value+range,"\t","\t","%.2e" % flow_network.flow_rate[eid])
    elif type==2:
        print("Velocity","\t",eid,"\t",value-range,"\t",value+range,"\t","\t","\t", "%.2e" % flow_network.rbc_velocity[eid])
