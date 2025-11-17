#!/usr/bin/env python

import igraph
import pickle
import numpy as np
import pyvista as pv

with open("./nw_output/graph_gen_process/test_1_semi_realistic_network_final.pkl", mode="rb") as f:
  g = pickle.load(f)

coords = np.asarray(g.vs["coords"])

edges = np.asarray(g.get_edgelist(), dtype=np.int64)
cells = np.hstack([np.full((edges.shape[0], 1), 2, dtype=np.int64), edges]).ravel()

mesh = pv.PolyData(coords, lines=cells)



for k in g.vs.attributes():
    if k == "coords":
        continue
    vals = np.asarray(g.vs[k], dtype=object)
    if vals.dtype == bool:
        vals = vals.astype(int)
    mesh.point_data[k] = vals

for k in g.es.attributes():
    vals = np.asarray(g.es[k], dtype=object)
    if vals.dtype == bool:
        vals = vals.astype(int)
    mesh.cell_data[k] = vals

mesh.save("test_1_graph.vtp", binary=True)


# pd = mesh.GetPointData()
# cd = mesh.GetCellData()
#
# import vtk
# arr = vtk.vtkDoubleArray()
# arr.SetName("diameter")
# for val in [1.2, 2.3, 3.4]:
#     arr.InsertNextValue(float(val))
# pd.AddArray(arr)
#
# sarr = vtk.vtkStringArray()
# sarr.SetName("type")
# for val in ["A", "B", "C"]:
#     sarr.InsertNextValue(val)
# cd.AddArray(sarr)
#
# from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
# writer = vtkXMLPolyDataWriter()
# writer.SetFileName("graph.vtp")
# writer.SetInputData(mesh)
# writer.SetDataModeToBinary()
# writer.Write()
