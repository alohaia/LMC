from lmc import nw, viz
from lmc.core import io, ops
from lmc.utils import ensure_dirs

import numpy as np

np.random.seed(666)

# mice = ["test1_hypevs", "test5_pbs"]
mice = ["11_hypEVs"]

for mouse in mice:
    # ensure_dirs([
    #     f"data/{mouse}/",
    #     f"output/{mouse}/refinement/",
    #     f"results/{mouse}/",
    #     f"figures/{mouse}/",
    # ])
    #
    # graph = io.create(data_xlsx=f"data/{mouse}/data.xlsx", name = f"Mouse {mouse}")
    #
    # graph_vor = nw.refine_DA_density(
    #     graph,
    #     ghp_boundary_offset=800,
    #     ghp_simplify_tolerance=10,
    #     nr_new_DAs_max=200,
    #     min_DA_DA_distance=200.,
    #     newDA_prior_target_density=11.5,
    #     nr_tries_max=1000,
    #     save_path=f"output/{mouse}/",
    #     save_refinement_steps=True,
    #     save_init_final_distribution=True
    # )
    # nw.add_AVs(graph, graph_vor)
    #
    # da_points = ops.filter_vs(graph, vtype=("DA root", "DA root added manually"))
    # graph_vor = ops.voronoi_tessalation(
    #     points=da_points,
    #     ghost_points=ops.gen_ghost_points(da_points),
    #     points_annot="DA points"
    # )
    #
    # is_successful = nw.connect_new_DA_roots(
    #     graph, min_dist_subpts=75, max_len_new_vessel=1000, distort_max=0.3
    # )
    #
    # io.save(graph, f"results/{mouse}/SA_network.xlsx")

    graph = io.load(f"results/{mouse}/SA_network.xlsx")

    fig, ax = viz.plot_graph(graph)
    fig.savefig(
        f"figures/{mouse}/SA_network.png",
        dpi=600,
        bbox_inches="tight"
    )

    da_candidates = np.array([0, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                            21, 22, 23, 24, 26, 27, 28, 29,
                            32, 33, 35, 36, 37, 39, 42, 44, 45, 46, 47, 48, 49,
                            50, 52, 54, 55, 56], dtype=np.int_)

    nw.add_da_trees_to_sa_graph(graph, da_candidates)

