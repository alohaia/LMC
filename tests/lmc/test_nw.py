import pytest
from pathlib import Path

import numpy as np

from igraph import Graph

from lmc.core import ops, io
from lmc import nw, viz

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def g_bare() -> Graph:
    graph = io.create(vertex_csv=DATA_DIR / "vertices.csv",
                      edge_csv=DATA_DIR / "edges.csv",
                      calc_length=False)
    return graph


@pytest.fixture(scope="session")
def g() -> Graph:
    graph = io.create(vertex_csv=DATA_DIR / "vertices.csv",
                      edge_csv=DATA_DIR / "edges.csv",
                      name="Test 1",
                      calc_length=True)
    ops.calc_es_length(graph)
    return graph

@pytest.fixture(scope="session")
def g_refined() -> Graph:
    return io.load(DATA_DIR / "output" / "graph_refined.xlsx")


def test_append_vertices(g_bare: Graph):
    vs_new = np.array([[1, 2], [3, 4], [1.4, 2.5]])
    nw.append_vertices(g_bare, vs_new, {"is_DA_root_added_manually": True})
    assert np.all(g_bare.vs["is_DA_root_added_manually"][-3:])


def test_check_attrs(g: Graph):
    nw.check_attrs(g)

    vs = g.get_vertex_dataframe()
    es = g.get_edge_dataframe()
    vs = vs.drop(columns="ACA_in")
    g_new = Graph.DataFrame(vertices=vs, edges=es)
    with pytest.raises(ValueError, match="ACA_in"):
        nw.check_attrs(g_new)


## for a1_script_sa_vasculature.py
@pytest.mark.slow
def test_refine_DA_density(g: Graph):
    # refine DA roots
    vor = nw.refine_DA_density(
        g,
        ghp_boundary_offset=800,
        ghp_simplify_tolerance=10,
        nr_new_DAs_max=200,
        min_DA_DA_distance=200.,
        newDA_prior_target_density=11.5,
        nr_tries_max=1000, # E = density * area_netwrk
        save_path="output/test_1/",
        save_refinement_steps=True,
        save_init_final_distribution=True
    )

    io.save(g, DATA_DIR / "output" / "graph_refined.xlsx")

    nw.add_AVs(
        g, vor,
        target_da_av_ratio=3,
        min_dist_2_DA = 100,
        min_dist_2_AV = 120,
        nr_tries_max=10000
    )

    viz.plot_graph(g)

    # visualize.visualize_DA_AV_roots_with_polygons(
    #     graph_SA, xys_new_DA_roots,
    #     # xy_DA_all, # xy_DA_roots + xy_DA_roots_new_points
    #     xy_AVs,
    #     polygon_vs_xy,
    #     show_MCA_ACA_root=False,
    #     title=title_plot,
    #     filepath="nw_output/DA_AV_locations/DA_AV_map.png"
    # )


def test_add_AVs(g_refined: Graph):
    # g_refined = nw.add_AVs(g_refined)
    pass


