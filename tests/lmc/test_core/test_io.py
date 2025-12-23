import pytest
from pathlib import Path

import numpy as np

from lmc import nw, viz
from lmc.core import ops, io
from igraph import Graph

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


def test_create_from_file():
    io.create(vertex_csv=DATA_DIR / "vertices.csv",
              edge_csv=DATA_DIR / "edges.csv", name="Test1")

    io.create("tests/lmc/data/vertices.csv", "tests/lmc/data/edges.csv",
              name="Test 1")

    with pytest.raises(FileNotFoundError):
        io.create("tests/lmc/data/vs.csv", "tests/lmc/data/es.csv",
                  name="Test 1")


# def test_create_from_data(g: Graph):
#     g_attrs, vs_attrs, es_attrs = io.get_data(g)
#     io.create_from_data(g_attrs, vs_attrs, es_attrs)


def test_save(g: Graph):
    io.save(g, DATA_DIR / "output" / "graph_init.xlsx")


def test_load(g: Graph):
    g_loaded = io.load(DATA_DIR / "output" / "graph_init.xlsx")

    vs = g.get_vertex_dataframe()
    es = g.get_edge_dataframe()
    vs_loaded = g_loaded.get_vertex_dataframe()
    es_loaded = g_loaded.get_edge_dataframe()

    assert ((vs == vs_loaded) | vs.isna()).all().all()
    # es.length maybe not exactly the same
    assert ((es.iloc[:, :-1] == es_loaded.iloc[:, :-1])
            | es.isna()).all().all()


