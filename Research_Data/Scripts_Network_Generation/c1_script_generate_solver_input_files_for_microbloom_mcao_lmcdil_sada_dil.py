import igraph
import numpy as np
import pandas as pd
import sys

from source.import_graph import check_graph_attributes_complete_nwgen
from source.export_graph import export_microbloom_setup_files

##################################################################
# Import the semi-realistic microvascular network
##################################################################

graph = igraph.Graph.Read_Pickle('nw_output/graph_gen_process/semi_realistic_network_final.pkl')

# Print the network summary and check the attributes
print graph.summary()
if not check_graph_attributes_complete_nwgen(graph):
    sys.exit()


##################################################################
# Set the diameter values for the testcases MCAo, MCAo & LMC-dil and MCAo & LMC/SA/DA-dil
# Write the network input files for microbloom
##################################################################

# Baseline diameters (pre-stroke).
# IMPORTANT: Here, the diameter values after tuning have to be used!!!
diameter_base = np.array(graph.es["diameter"])
graph.es["diameter_base"] = diameter_base

# MCAo diameters
is_stroke = np.array(graph.es['is_stroke']) > 0  # Identify edge with MCA occlusion
diameter_mcao = np.copy(diameter_base)
diameter_mcao[is_stroke] = diameter_mcao[is_stroke] / 10.  # Constrict stroke edge by 90%
graph.es["diameter_mcao"] = diameter_mcao

# MCAo and LMC-dil diameters
is_collateral = np.array(graph.es['is_collateral']) > 0
collateral_post_diam = np.array(graph.es['diam_post_exp'])[is_collateral]
diameter_lmcdil = np.copy(diameter_mcao)
diameter_lmcdil[is_collateral] = collateral_post_diam
graph.es["diameter_lmcdil"] = diameter_lmcdil

# MCAo and LMC/SA/DA-dil diameters
types = np.array(graph.es["type"])
is_sa_da = np.logical_or(types == 0, types == 2)  # 0: surface artery, 2: descending artery
diameter_lmcsada_dil = np.copy(diameter_lmcdil)
diameter_lmcsada_dil[is_sa_da] = diameter_lmcdil[is_sa_da] * 1.1  # fixed 10 % dilation of SAs & DAs compared to baseline
diameter_lmcsada_dil[is_collateral] = diameter_lmcdil[is_collateral]  # Collaterals are dilated according to measurements
graph.es["diameter_lmcsada_dil"] = diameter_lmcsada_dil

##################################################################
# Export setup files for microbloom
##################################################################

export_microbloom_setup_files(graph, diameter_attribute='diameter_base',
                              p_in_cow_mmHg=100., p_out_av_mmHg=10.,
                              write_inverse_model_data=False,
                              path="nw_output/setup_microbloom_stroke_scenarios/01_Baseline/")

export_microbloom_setup_files(graph, diameter_attribute='diameter_mcao',
                              p_in_cow_mmHg=100., p_out_av_mmHg=10.,
                              write_inverse_model_data=False,
                              path="nw_output/setup_microbloom_stroke_scenarios/02_MCAo/")

export_microbloom_setup_files(graph, diameter_attribute='diameter_lmcdil',
                              p_in_cow_mmHg=100., p_out_av_mmHg=10.,
                              write_inverse_model_data=False,
                              path="nw_output/setup_microbloom_stroke_scenarios/03_MCAo_and_LMC_dil/")

export_microbloom_setup_files(graph, diameter_attribute='diameter_lmcsada_dil',
                              p_in_cow_mmHg=100., p_out_av_mmHg=10.,
                              write_inverse_model_data=False,
                              path="nw_output/setup_microbloom_stroke_scenarios/04_MCAo_and_LMC_SA_DA_dil/")

##################################################################
# Export setup files for microbloom related to the distensibility of blood vessels
##################################################################

is_stroke_or_collateral = np.logical_or(is_stroke, is_collateral)

# IMPORTANT: Here the diameter values after tuning have to be used!!!
diameters_baseline_si = np.array(graph.es["diameter_base"]) * 1e-6

wall_thickness = diameters_baseline_si * .1
e_modulus = np.zeros(graph.ecount())

is_artery = (types <= 2)
e_modulus[is_artery] = 1.6e5  # Types -1, 0, 2 (Arteries)
e_modulus[np.logical_and(is_artery, diameters_baseline_si >= 15e-6)] = 2.6e5
e_modulus[np.logical_and(is_artery, diameters_baseline_si >= 20e-6)] = 6.3e5
e_modulus[types == 3] = 3.88e5  # Type 3 (Veins)
e_modulus[types == 4] = 3.70e5  # Type 4 (Capillary)

# Generate distensibility parameter file for each vessel of the test case with MCAo
df_ocldist = pd.DataFrame()
df_ocldist["eid_distensibility"] = np.arange(graph.ecount())[np.logical_not(is_stroke)]
df_ocldist["e_modulus"] = e_modulus[np.logical_not(is_stroke)]
df_ocldist["wall_thickness"] = wall_thickness[np.logical_not(is_stroke)]
df_ocldist.to_csv("nw_output/setup_microbloom_stroke_scenarios/02_MCAo/distensibility_parameters.csv", index=False)

# Generate distensibility parameter file for each vessel of the test case with MCAo and lmc dilation
df_dildist = pd.DataFrame()
df_dildist["eid_distensibility"] = np.arange(graph.ecount())[np.logical_not(is_stroke_or_collateral)]
df_dildist["e_modulus"] = e_modulus[np.logical_not(is_stroke_or_collateral)]
df_dildist["wall_thickness"] = wall_thickness[np.logical_not(is_stroke_or_collateral)]
df_dildist.to_csv("nw_output/setup_microbloom_stroke_scenarios/03_MCAo_and_LMC_dil/distensibility_parameters.csv", index=False)

# Generate distensibility parameter file for each vessel of the test case with MCAo and lmc/sa/da dilation
is_stroke_or_collateral_or_sa_da = np.logical_or(is_stroke_or_collateral, is_sa_da)
df_dil_sa_dist = pd.DataFrame()
df_dil_sa_dist["eid_distensibility"] = np.arange(graph.ecount())[np.logical_not(is_stroke_or_collateral_or_sa_da)]
df_dil_sa_dist["e_modulus"] = e_modulus[np.logical_not(is_stroke_or_collateral_or_sa_da)]
df_dil_sa_dist["wall_thickness"] = wall_thickness[np.logical_not(is_stroke_or_collateral_or_sa_da)]
df_dil_sa_dist.to_csv("nw_output/setup_microbloom_stroke_scenarios/04_MCAo_and_LMC_SA_DA_dil/distensibility_parameters.csv", index=False)

# File with the baseline diameters at the baseline pressure (for reference)
df_diameters_base = pd.DataFrame()
df_diameters_base["d_base"] = diameters_baseline_si
df_diameters_base.to_csv("nw_output/setup_microbloom_stroke_scenarios/02_MCAo/baseline_diameters.csv", index=False)
df_diameters_base.to_csv("nw_output/setup_microbloom_stroke_scenarios/03_MCAo_and_LMC_dil/baseline_diameters.csv", index=False)
df_diameters_base.to_csv("nw_output/setup_microbloom_stroke_scenarios/04_MCAo_and_LMC_SA_DA_dil/baseline_diameters.csv", index=False)

################
# Write to file
################

graph.write_pickle('nw_output/setup_microbloom_stroke_scenarios/network_stroke_scenarios.pkl')
