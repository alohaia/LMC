import pickle

import numpy as np

from lmc import config, nw, viz
from lmc.core import io
from lmc.utils import ensure_dirs

np.random.seed(config.random_seed)


for mouse in config.mice:
    ensure_dirs([
        f'output/{mouse}/refinement/',
        f'results/{mouse}/',
        f'figures/{mouse}/',
    ])

    with open(f'results/{mouse}/_CoW_added.pkl', 'rb') as f:
        graph = pickle.load(f)

    # 1. Refine DA roots and add AV roots
    print(f'=> "{mouse}"\tRefine SA network...')

    print(f'==> "{mouse}"\tAdd DA roots...')
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
    print(f'==> "{mouse}"\tAdd AV roots (outflow points)...')
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

    # 2. Add penetrating vessel trees
    print(f'=> "{mouse}"\tAdd penetrating trees...')
    # with open(f'results/{mouse}/_SA_network.pkl', 'rb') as f:
    #     graph = pickle.load(f)
    nw.add_pt_trees(graph, tree_type=['DA', 'AV'])
    graph.write_pickle(f'results/{mouse}/_Penetrating_trees_added.pkl')
    print(f'=> "{mouse}"\tAdd penetrating trees. Done')

    # 3. Add capillary bed
    print(f'=> "{mouse}"\tAdd Caps bed...')
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

    # # 4. TODO: crop redundant vertices and edges {{{
    # print(f'=> "{mouse}"\tCrop redundant parts...')
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

    # 5. Manually narrow stroke vessel
    # c1_script_generate_solver_input_files_for_microbloom_mcao_lmcdil_sada_dil.py:l34
    print(f'=> "{mouse}"\tConstrict stroke vessel...')
    stroke_edge = graph.es(is_stroke=True)
    assert len(stroke_edge) == 1
    stroke_edge['diameter_mcao0h'] = stroke_edge['diameter'][0] * 0.1
    stroke_edge['diameter_mcao1h'] = stroke_edge['diameter'][0] * 0.1
    print(f'=> "{mouse}"\tConstrict stroke vessel. Done')

    # 6. other adjustments for flow model
    # set boundary
    graph.vs(is_CoW_in_eq=True)['type'] = 'CoW in'
    # 1 for pressure, Nan for non-boundary vertex
    graph.vs(type_in=('CoW in', 'AV root'))['boundaryType'] = 1
    # convert unit
    # mmHg -> Pa
    graph.vs(type='AV root')['boundaryValue'] = 10 * 133.322
    graph.vs(type=('CoW in'))['boundaryValue'] = 100 * 133.322
    #########################################################################
    # XXXXXXXXXX micrometer -> meter XXXXXXXXXX                             #
    # DO ONT convert, or system_matrix will be not able to be solved.       #
    # See length_scale.py for understading final values.                    #
    #########################################################################
    graph.vs['coords'] = \
        np.array((graph.vs['x'], graph.vs['y'], graph.vs['z'])).T
    # graph.es[_diameter_attr] = np.array(graph.es[_diameter_attr]) * 10e-6
    # graph.es['length'] = np.array(graph.es['length']) * 10e-6

    # 7. Save final results
    print(f'=> "{mouse}"\tSave final results...')
    graph.write_pickle(f'results/{mouse}/semi_realistic_vessel_network.pkl')
    # io.save(graph, f'results/{mouse}/semi_realistic_vessel_network.xlsx')
    io.write_vtk(graph, f'results/{mouse}/semi_realistic_vessel_network.vtu')
    print(f'=> "{mouse}"\tSave final results. Done')


