#!/usr/bin/env python

import pickle
import json

import numpy as np

class NumpyEncoder(json.JSONEncoder):
  def default(self, o):
    if isinstance(o, np.ndarray):
      return o.tolist()
    if isinstance(o, (np.float32, np.float64)):
      return float(o)
    if isinstance(o, (np.int32, np.int64)):
      return int(o)
    return super().default(o)

with open("./example_data/Research_Data/Scripts_Network_Generation/surface_artery_networks/C57BL6_1/edgesDict.pkl", 'rb') as f:
  data_edge = pickle.load(f)

print(data_edge.keys())

with open("./example_data/Research_Data/_test/C57BL6_1_edgesDict.json", mode="w") as f:
  json.dump(data_edge, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)



with open("./example_data/Research_Data/Scripts_Network_Generation/surface_artery_networks/C57BL6_1/verticesDict.pkl", 'rb') as f:
  data_vrt = pickle.load(f)

print(data_vrt.keys())

with open("./example_data/Research_Data/_test/C57BL6_1_verticesDict.json", mode="w") as f:
  json.dump(data_vrt, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
