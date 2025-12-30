from lmc.core import io
import numpy as np
import pickle

graph = io.load("./results/11_hypEVs/sem_irealistic_vessel_network.xlsx")

graph.vs["coords"] = np.array((graph.vs["x"], graph.vs["y"], graph.vs["z"])).T
vs_DA_roots = graph.vs(type_in=("DA root", "DA root added manually"))
vs_AV_roots = graph.vs(type_in=("AV root",))
vs_DA_roots["boundaryType"] = "1"
vs_AV_roots["boundaryType"] = "1"


graph.vs["boundaryValue"] =

