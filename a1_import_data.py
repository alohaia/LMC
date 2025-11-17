#!/usr/bin/env python

import time
import datetime

import pandas as pd

import igraph


vs_data = pd.read_csv("../data/test 1/vertices.csv")
es_data = pd.read_csv("../data/test 1/edges.csv")

g = igraph.Graph((es_data[["from", "to"]] - 1).to_numpy().tolist())

ts = time.time()
g['network_gen_date'] = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
g['Mouse'] = "Test 1"

vs_coords = vs_data.loc[:, ["X", "Y"]].to_numpy()
vs_coords = np.hstack([vs_coords, np.zeros((vs_coords.shape[0], 1))])
g.vs["coords"] = vs_coords
g.vs["MCA_in"] = vs_data.MCA_in
g.vs["ACA_in"] = vs_data.ACA_in
g.vs["is_DA_startingPt"] = vs_data.is_DA_startingPt # FIXME
g.vs["is_DA_startingPt_added_manually"] = np.zeros(g.vcount(), dtype=int)
g.vs['is_AV_root'] = np.zeros(g.vcount(), dtype=int)
g.vs['is_connected_2caps'] = np.zeros(g.vcount(), dtype=int)
g.vs['COW_in'] = np.zeros(g.vcount(),dtype=int)


g.es['is_stroke'] = es_data.is_stroke
g.es["diameter"] = es_data.diameter
g.es["isCollateral"] = es_data.isCollateral
g.es['type'] = es_data.type
g.es["added_manually"] = es_data.added_manually

# FIXME
g.es["index_exp"] = es_data.index_exp
g.es["diam_pre_exp"] = es_data.diam_pre_exp
g.es["diam_post_exp"] = es_data.diam_post_exp
g.es["vRBC_pre_exp"] = es_data.vRBC_pre_exp
g.es["vRBC_post_exp"] = es_data.vRBC_post_exp


lengths = []
for e in g.es:
    v1, v2 = e.tuple
    p1, p2 = g.vs[v1]["coords"], g.vs[v2]["coords"]
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    lengths.append(dist)
g.es["length"] = lengths


igraph.plot(
    g,
    layout=[arr[:2] for arr in g.vs["coords"]],
    vertex_size=4,
    vertex_color="lightblue",
    edge_color="gray",
    bbox=(600, 600),
    margin=30
)

graph_surface_artery = g
