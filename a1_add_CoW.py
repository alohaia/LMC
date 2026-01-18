import matplotlib.ticker as mticker
import numpy as np
from igraph import Graph

from lmc import config, nw, viz
from lmc.core import io, ops
from lmc.utils import ensure_dirs

np.random.seed(config.random_seed)

if config.is_cow_added:
    print(
        'ACA:\n'
        'etype\tlength\tdiameter\n'
        '-11\t2250\t65\n'
        '-12\t2250\t50\n'
        '-13\t1000\t40\n'
        '-15\t500\t40\n'
        '\nMCA:\n'
        'etype\tlength\tdiameter\n'
        '-21\t2250\t94\n'
        '-22\t2250\t92\n'
        '-23\t950\t60\n'
        '-24\t50\t60\n'
        '-25\t450\t55\n'
        '-26\t500\t55\n'
    )

def graph_preprocess(graph: Graph) -> None:
    # z coordinate
    graph.vs['z'] = np.zeros(graph.vcount(), dtype=np.float64)

    # type of vertices and edges {{{
    if 'type' in graph.vs.attributes():
        vtypes = np.array(graph.vs['type'])
        vtypes[np.isnan(vtypes)] = 'PA point'
        vtypes[np.equal(graph.vs["is_DA_root"], True)] = 'DA root'
        graph.vs['type'] = vtypes
    else:
        graph.vs['type'] = 'PA point'
        graph.vs(is_DA_root_eq=True)['type'] = 'DA root'

    if config.is_cow_added:
        etypes = graph.es['type']
        graph.es['length'] = [
            np.nan
                if np.isnan(etypes[ie]) else config.es_length_map[etypes[ie]]
            for ie in range(graph.ecount())
        ]
        for dattr in config.diameter_attrs:
            graph.es[dattr] = [
                graph.es[ie][dattr]
                    if np.isnan(etypes[ie]) else config.es_diameter_map[etypes[ie]]
                for ie in range(graph.ecount())
            ]

    graph.es['type'] = 'PA'
    # }}}

    # calculate edge lengths
    if 'length' in graph.es.attributes():
        es_length = np.array(graph.es['length'])
        es_length_calc = ops.calc_es_length(graph)
        es_length[np.isnan(es_length)] = es_length_calc[np.isnan(graph.es['length'])]
        graph.es['length'] = es_length
    else:
        graph.es['length'] = ops.calc_es_length(graph)

    if 'is_added_manually' not in graph.es.attributes():
        graph.es['is_added_manually'] = False

    if 'is_added_manually' not in graph.vs.attributes():
        graph.vs['is_added_manually'] = False

    nw.check_attrs(graph,
                   'after_adding_CoW' if config.is_cow_added else 'before_adding_CoW')


for mouse in config.mice:
    ensure_dirs([
        f'results/{mouse}/',
        f'figures/{mouse}/',
    ])

    print(f'=> "{mouse}"\tLoad and pre-process data ...')
    graph = io.create(
        data_xlsx=f'data/{mouse}/data.xlsx',
        name=f'Mouse {mouse}',
        convert_bool_attrs=True,
        use_vids=False
    )
    graph_preprocess(graph)
    print(f'=> "{mouse}"\tLoad and pre-process data. Done.',)

    # plot vertices for the convience of adding CoW
    fig, ax = viz.plot_graph(graph)

    vs_mca_in = graph.vs(is_MCA_in_eq=True)
    vs_aca_in = graph.vs(is_ACA_in_eq=True)
    ax.scatter(
        label="MCA inflow point",
        x=vs_mca_in["x"], y=vs_mca_in["y"],
        s=150, marker="o", edgecolors="black", c="red"
    )
    ax.scatter(
        label="ACA inflow point",
        x=vs_aca_in["x"], y=vs_aca_in["y"],
        s=150, marker="o", edgecolors="black", c="blue"
    )
    for i in range(graph.vcount()):
        ax.text(
            x=graph.vs[i]["x"], y=graph.vs[i]["y"],
            s=graph.vs[i]["name"]
        )

    ax.axis("on")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(200))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(200))
    ax.tick_params(axis="x", rotation=45)

    ax.grid(True)
    ax.margins(x=0.4, y=0.4)
    ax.legend()

    fig.set_size_inches(10, 10)

    fig.savefig(
        f'figures/{mouse}/_adding_CoW.png',
        dpi=600,
        # bbox_inches='tight'
    )

    if config.is_cow_added:
        graph.write_pickle(f"results/{mouse}/_CoW_added.pkl")


