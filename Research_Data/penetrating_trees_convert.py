import pickle
import pdb
import pandas as pd

with open("Research_Data/Scripts_Network_Generation/penetrating_trees/arteries/0_edgesDict.pkl", "rb") as e:
    es = pickle.load(e)

with open("Research_Data/Scripts_Network_Generation/penetrating_trees/arteries/0_verticesDict.pkl", "rb") as v:
    vs = pickle.load(v)

pdb.set_trace()

es = pd.DataFrame(es)
vs = pd.DataFrame(vs)

pdb.set_trace()
