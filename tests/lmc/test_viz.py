import pytest
from pathlib import Path

from igraph import Graph

from lmc.core import ops, io
from lmc import viz

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def g_refined() -> Graph:
    return io.load(DATA_DIR / "output" / "graph_refined.xlsx")

def test_plot_graph(g_refined: Graph):
    viz.plot_graph(g_refined)

def test_plot_voronoi(g_refined):
    points = ops.filter_vs(g_refined, "is_DA_root", z=False)
    ght_points = ops.gen_ghost_points(points, 800, 10)
    vor = ops.voronoi_tessalation(points, ght_points, "DA root")
    viz.plot_voronoi(vor)
