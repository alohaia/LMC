from sys import breakpointhook
from igraph import Graph
from lmc import nw, viz
from lmc.core import io, ops
from lmc.utils import ensure_dirs

import numpy as np
import pickle

np.random.seed(666)

# rng = np.random.default_rng(666);
# rng.random()

# mice = ['11_hypEVs']
mice = ['1_hypEVs', '5_PBS', '8_norEVs', '11_hypEVs']

def graph_preprocess(graph: Graph) -> None:
    ## z coordinate
    graph.vs['z'] = np.zeros(graph.vcount(), dtype=np.float64)

    ## type of vertices and edges
    graph.vs['type'] = 'PA point'
    graph.vs(is_DA_root=True)['type'] = 'DA root'
    graph.es['type'] = 'PA'

    ## calculate edge lengths
    graph.es['length'] = ops.calc_es_length(graph)


for mouse in mice:
    ensure_dirs([
        f'data/{mouse}/',
        f'output/{mouse}/refinement/',
        f'results/{mouse}/',
        f'figures/{mouse}/',
    ])

    # 1. Load data
    print(f'=> "{mouse}"\tLoad data ...')
    graph = io.create(
        data_xlsx=f'data/{mouse}/data.xlsx',
        name=f'Mouse {mouse}',
        convert_bool_attrs=True,
        use_vids=False
    )
    graph_preprocess(graph)
    print(f'=> "{mouse}"\tLoad data. Done.',)

    # 2. Refine DA roots and add AV roots
    print(f'=> "{mouse}"\tRefine SA network ...')

    print(f'==> "{mouse}"\tAdd DA roots ...')
    graph_vor = nw.refine_DA_density(
        graph,
        ghp_boundary_offset=800,
        ghp_simplify_tolerance=10,
        nr_new_DAs_max=200,
        min_DA_DA_distance=200.,
        newDA_prior_target_density=11.5,
        nr_tries_max=1000,
        save_path=f'output/{mouse}/',
        save_refinement_steps=True,
        save_init_final_distribution=True
    )
    print(f'==> "{mouse}"\tAdd DA roots. Done.')
    print(f'==> "{mouse}"\tAdd AV roots (outflow points) ...')
    nw.add_AVs(graph, graph_vor)
    nw.connect_new_DA_roots(
        graph, min_dist_subpts=75, max_len_new_vessel=1000, distort_max=0.3
    )
    print(f'==> "{mouse}"\tAdd AV roots (outflow points). Done')

    # Save results
    graph.write_pickle(f'results/{mouse}/_SA_network.pkl')
    fig, ax = viz.plot_graph(graph)
    fig.savefig(
        f'figures/{mouse}/SA_network.png',
        dpi=600,
        bbox_inches='tight'
    )

    print(f'=> "{mouse}"\tRefine SA network. Done.')

    # 3. Add penetrating vessel trees
    print(f'=> "{mouse}"\tAdd penetrating trees ...')
    # with open(f'results/{mouse}/_SA_network.pkl', 'rb') as f:
    #     graph = pickle.load(f)
    nw.add_pt_trees(graph, tree_type=['DA', 'AV'])
    graph.write_pickle(f'results/{mouse}/_Penetrating_trees_added.pkl')
    print(f'=> "{mouse}"\tAdd penetrating trees. Done')

    # 4. Add capillary bed
    print(f'=> "{mouse}"\tAdd Caps bed ...')
    # with open(f'results/{mouse}/_Penetrating_trees_added.pkl', 'rb') as f:
    #     graph = pickle.load(f)
    nw.add_capillary_bed(
        graph,
        z_min_caps=25.0,
        frame_bounding_box=50.0,
        distance_between_cap_vs=45.0,
        l_vessel=62.0,
        d_vessel=4.0,
        perturb_vs_frac=0.8
    )
    graph.write_pickle(f'results/{mouse}/_Caps_bed_added_method.pkl')
    print(f'=> "{mouse}"\tAdd Caps bed. Done')

    # # 5. TODO: crop redundant vertices and edges {{{
    # print(f'=> "{mouse}"\tCrop redundant parts ...')
    # with open(f'results/{mouse}/_Caps_bed_added_method_3.pkl', 'rb') as f:
    #     graph = pickle.load(f)
    #
    # breakpoint()
    # nw.remove_redundant_vessels(graph)
    #
    # print('==> Graph is connected:', graph.is_connected())
    # # If the graph is not connected: manually remove the smaller components. Show the components:
    # # print graph.components()
    # print(f'=> "{mouse}"\tCrop redundant parts. Done.')
    # # }}}

    # 6. Save final results
    print(f'=> "{mouse}"\tSave final results ...')
    graph.write_pickle(f'results/{mouse}/semi_realistic_vessel_network.pkl')
    # io.save(graph, f'results/{mouse}/semi_realistic_vessel_network.xlsx')
    io.write_vtk(graph, f'results/{mouse}/semi_realistic_vessel_network.vtu')
    print(f'=> "{mouse}"\tSave final results. Done')


