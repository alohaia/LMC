#!/usr/bin/env python

import pickle
import subprocess
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from igraph import Graph

from lmc import config, viz
from lmc.core import io, ops

conditions = {
    'baseline',
    'MCAO0h',
    'MCAO1h'
}

for m in config.mice:
    print(f'>>>>> Processing {m} <<<<<')

    with open(f'results/{m}/semi_realistic_vessel_network.pkl', 'rb') as f:
        graph = pickle.load(f)

    ## Data of entire networks
    edges = graph.get_edge_dataframe()
    vertices = graph.get_vertex_dataframe()

    for cond in conditions:
        prefix = f'{cond}.'

        with open(f'microBlooM/result/{m}/{m}_{cond}_FlowNetwork.pkl', 'rb') as f:
            g = pickle.load(f)

        es = g.get_edge_dataframe()
        vs = g.get_vertex_dataframe()
        # es = es.rename(columns={'diameter': f'diameter_{cond}'}).add_prefix(prefix)
        es = es.add_prefix(prefix)
        vs = vs.add_prefix(prefix)

        edges = pd.concat([edges, es], axis=1)
        vertices = pd.concat([vertices, vs], axis=1)

    edges_lmc = edges[edges.loc[:, 'is_collateral'] == True]
    if len(edges_lmc) >= 1:
        i_lmc_endpts = pd.concat([edges_lmc.loc[:, 'source'], edges_lmc.loc[:, 'target']])
        i_lmc_endpts = i_lmc_endpts[~(i_lmc_endpts.duplicated(keep=False))]
        lmc_endpts = np.array(vertices.iloc[i_lmc_endpts].loc[:, ('x', 'y')])

        conject_list = np.array(edges.loc[:, ('source', 'target')])
        vs_coords = np.array(vertices.loc[:, ('x', 'y')])
        edge_endpts = vs_coords[conject_list]
        edge_midpts = np.mean(edge_endpts, axis=1)

        ## Edges to LMC end points
        edges['closest_dist_midpoint_to_LMC (μm)'] = np.min(
            np.linalg.norm(
                edge_midpts[np.newaxis, :, :] - lmc_endpts[:, np.newaxis, :],
                axis=2
            ),
            axis=0
        )
        # LMC, edge, which edge end, xy
        edges['closest_dist_endpoint_to_LMC (μm)'] = np.min(
            np.linalg.norm(
                edge_endpts[np.newaxis,] - lmc_endpts[:, np.newaxis, np.newaxis, :],
                axis=3
            ),
            axis=(0,2)
        )

        ## Vertices to LMC end points
        vertices['closest_dist_to_LMC (μm)'] = np.min(
            np.linalg.norm(
                vs_coords[np.newaxis, :, :] - lmc_endpts[:, np.newaxis, :],
                axis=2
            ),
            axis=0
        )
    else:
        edges['closest_dist_to_LMC (μm)'] = np.inf
        vertices['closest_dist_to_LMC (μm)'] = np.inf

    edges.to_csv(f'microBlooM/result/{m}/{m}_edges.csv')
    vertices.to_csv(f'microBlooM/result/{m}/{m}_vertices.csv')

    ## Data of SA networks

    # edges = pd.read_csv(f'microBlooM/result/{m}/{m}_edges.csv', index_col='edge ID')
    # vertices = pd.read_csv(f'microBlooM/result/{m}/{m}_vertices.csv',
    #                        index_col='vertex ID')

    graph_int = Graph.DataFrame(edges=edges, vertices=vertices,
                                directed=False, use_vids=True)

    graph_sa = graph_int.subgraph_edges(
        graph_int.es.select(is_added_manually_ne=True, type_eq='PA',
                            is_collateral_ne=True),
        delete_vertices=True
    )
    graph_sa.es['origin'] = ""
    try:
        for v in graph_sa.vs(is_MCA_in_eq=True):
            reachable_vs = graph_sa.subcomponent(v, mode="ALL")
            reachable_es = graph_sa.es.select(_within=reachable_vs)
            assert np.all(np.equal(reachable_es['origin'], ''))
            reachable_es['origin'] = 'MCA'
        for v in graph_sa.vs(is_ACA_in_eq=True):
            reachable_vs = graph_sa.subcomponent(v, mode="ALL")
            reachable_es = graph_sa.es.select(_within=reachable_vs)
            assert np.all(np.equal(reachable_es['origin'], ''))
            reachable_es['origin'] = 'ACA'

        assert not np.any(np.equal(graph_sa.es['origin'], ''))
    except AssertionError as e:
        traceback.print_exc()
        print(str(e.args))
        breakpoint()

    fig, ax = viz.plot_sa(graph_sa)
    fig.savefig(f'microBlooM/result/{m}/{m}_SA_network.png', dpi=600)

    es_sa = graph_sa.get_edge_dataframe()
    fr_baseline_sa = es_sa['baseline.flow_rate']
    es_sa['rel.flow_rate_change.baseline2MCAO0h'] = \
        (es_sa['MCAO0h.flow_rate'] - fr_baseline_sa) / fr_baseline_sa
    es_sa['rel.flow_rate_change.baseline2MCAO1h'] = \
        (es_sa['MCAO1h.flow_rate'] - fr_baseline_sa) / fr_baseline_sa

    es_sa.to_csv(f'microBlooM/result/{m}/{m}_edges_sa.csv')

    vs_sa = graph_sa.get_vertex_dataframe()
    vs_sa.to_csv(f'microBlooM/result/{m}/{m}_vertices_sa.csv')

