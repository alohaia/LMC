import pytest
from pathlib import Path

from lmc import viz
from lmc.core import ops, io

from igraph import Graph

from lmc.types import VoronoiExt

DATA_DIR = Path(__file__).parent.parent / "data"

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
def vor(g: Graph) -> VoronoiExt:
    points = ops.filter_vs(g, "is_DA_root")
    ght_points = ops.gen_ghost_points(points, 800, 10)
    return ops.voronoi_tessalation(points, ght_points, "DA root")


def test_calc_es_length(g_bare: Graph):
    assert not "length" in g_bare.es.attributes()
    g_bare.es["length"] = ops.calc_es_length(g_bare)
    assert "length" in g_bare.es.attributes()


def test_get_DA_regions(vor: VoronoiExt):
    ops.get_DA_regions(vor)


def test_voronoi_tessalation(g: Graph):
    points = ops.filter_vs(g, "is_DA_root")
    ght_points = ops.gen_ghost_points(points, 800, 10)
    vor = ops.voronoi_tessalation(points, ght_points, "DA root")
    viz.plot_voronoi(vor)
    r1, r2 = ops.get_areas_voronoi_polygons(vor)
    assert r1[12] == 71208.80057766847
    assert r2[0] == 256235.09341127612

