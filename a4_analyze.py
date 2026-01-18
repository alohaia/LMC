#!/usr/bin/env python

import pickle

import pandas as pd

from lmc import config

conditions = {
    'baseline',
    'MCAO0h',
    'MCAO1h'
}

for m in config.mice:
    with open(f'results/{m}/semi_realistic_vessel_network.pkl', 'rb') as f:
        graph = pickle.load(f)

    edges = graph.get_edge_dataframe()
    vertices = graph.get_vertex_dataframe()

    for cond in conditions:
        prefix = f'{cond}.'

        with open(f'microBlooM/result/{m}/{m}_{cond}_FlowNetwork.pkl', 'rb') as f:
            g = pickle.load(f)

        es = g.get_edge_dataframe()
        vs = g.get_vertex_dataframe()
        # es = es.rename(columns={'diameter': f'diameter_{cond}'}).add_prefix(prefix)
        es = es.rename(columns={'diameter': f'diameter_{cond}'}).add_prefix(prefix)
        vs = vs.add_prefix(prefix)

        edges = pd.concat([edges, es], axis=1)
        vertices = pd.concat([vs, vs], axis=1)

    edges.to_csv(f'microBlooM/result/{m}/{m}_edges.csv')
    vertices.to_csv(f'microBlooM/result/{m}/{m}_vertices.csv')
