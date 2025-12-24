"""Base data reading, writing and conversion operations."""

from typing import Literal, Union
from pathlib import Path

from matplotlib.pyplot import uninstall_repl_displayhook
from pandas.core.series import astype_is_view

from lmc import utils
from lmc.core import ops, io
from lmc.types import Vertices
from lmc import config

import numpy as np
import pandas as pd
from igraph import Graph

import pickle


def create_from_file(
    # data_xlsx
    data_xlsx: Union[Path, str] = "",
    # dir
    dir: Union[Path, str] = "",
    # vertex_csv&edge_csv
    vertex_csv: Union[Path, str] = "",
    edge_csv: Union[Path, str] = "",
    name: str = "",
    as_is: bool = False,
    calc_length: bool = True,
    use_vids: bool = False
) -> Graph:
    """Create an igraph Graph object from vertices and edges CSV data.

    data_xlsx > dir > vertex_csv&edge_csv

    Args:
        data_xlsx: Path to the xlsx file containing two sheets with names of
            "vertices" and "edges".
        dir: Directory containing "edges.csv" and "vertices.csv".
        vertex_csv: Path to the CSV file containing the vertex data.
        edge_csv: Path to the CSV file containing the edge data.
        name: Value of graph-level `"name"` attribute.
        as_is: Wheter to create graph AS IS (only containing auto-genrated
            graph-level attributes and original data).
        calc_length: Whether to automatically caculate lengthes of edges.
        use_vids: Whether to interpret first two columns of edge data as vertex
            indices or vertex names specified by first column of vertex data.

    Returns:
        `igraph.Graph` object constructed from given data.
    """
    from time import time
    from datetime import datetime as dt

    if data_xlsx != "":
        dfs = utils.read_xlsx(data_xlsx)
        vs, es = dfs["vertices"], dfs["edges"]
    else:
        if dir != "":
            vcsv = Path(dir) / "vertices.csv"
            ecsv = Path(dir) / "edges.csv"
        else:
            vcsv = vertex_csv
            ecsv = edge_csv

        vs = pd.read_csv(vcsv)
        es = pd.read_csv(ecsv)
    # }}}

    # graph-level attributes
    gattrs = {
        "gen_date": dt.fromtimestamp(time()).strftime("%Y-%m-%d %H:%M:%S")
    }
    if name != "":
        gattrs["name"] = name

    return create_from_data(
        vs_attrs_df=vs,
        es_attrs_df=es,
        graph_attrs=gattrs,
        as_is=as_is,
        calc_length=calc_length,
        use_vids=use_vids
    )

## alias
create = create_from_file


def create_from_data(
    vs_attrs_df: pd.DataFrame,
    es_attrs_df: pd.DataFrame,
    graph_attrs: dict = {},
    as_is: bool = False,
    calc_length: bool = True,
    use_vids: bool = False
) -> Graph:
    """Create an igraph Graph object from vertices and edges `DataFrame` data.

    Args:
        vs_attrs_df: `DataFrame` of vertex attributes.
        es_attrs_df: `DataFrame` of edge attributes.
        graph_attrs: Graph-level attributes.
        as_is: Wheter to create graph AS IS (only containing auto-genrated
            graph-level attributes and original data).
        calc_length: Whether to automatically caculate lengthes of edges.
        use_vids: Whether to interpret first two columns of edge data as vertex
            indices or vertex names specified by first column of vertex data.

    Returns:
        `igraph.Graph` object constructed from given data.
    """
    g = Graph.DataFrame(vertices=vs_attrs_df, edges=es_attrs_df,
                        directed=False, use_vids=use_vids)
    # graph-level attributes {{{
    for k,v in graph_attrs.items():
        g[k] = v
    # }}}

    # return AS IS.
    if as_is:
        return g

    # additional processings {{{
    ## z coordinate
    if "z" not in g.vs.attributes():
        g.vs["z"] = np.zeros(g.vcount(), dtype=np.float64)

    ## convert attributes starting with "is_" and containing only 0, 1, nan to
    ## bool
    for a in ("vs", "es"):
        vore = getattr(g, a)
        for attr in vore.attributes():
            if attr.startswith("is_") and all(
                    map(lambda x: x in [0, 1] or np.isnan(x), vore[attr])):
                vore[attr] = [a == 1 for a in vore[attr]]

    ## type of vertices and edges
    g.vs["type"] = np.where(g.vs["is_DA_root"], "DA root", "")
    g.es["type"] = 0    # edge type: 0-PA; 1-PV; 2-DA; 3-AV; 4-C

    ## calculate edge lengths
    if calc_length:
        g.es["length"] = ops.calc_es_length(g)
    # }}}

    return g


def get_data(g: Graph) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Extract data from an `igraph.Graph` object."""
    g_attrs = { attr: g[attr] for attr in g.attributes() }
    vs_attrs_df = g.get_vertex_dataframe()
    es_attrs_df = g.get_edge_dataframe()
    return g_attrs, vs_attrs_df, es_attrs_df


def save(g: Graph, path: Union[Path, str]) -> None:
    """Save data of a graph to a xlsx file."""
    g_attrs, vs_attrs, es_attrs = get_data(g)
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({k: [v] for k,v in g_attrs.items()}) \
            .to_excel(writer, sheet_name="graph", index=False)
        vs_attrs.to_excel(writer, sheet_name="vertices", index=False)
        es_attrs.to_excel(writer, sheet_name="edges", index=False)


def load(path: Union[Path, str]):
    """Load graph from a xlsx file."""
    dt = pd.read_excel(path, sheet_name=None)
    return create_from_data(
        graph_attrs=dt["graph"].to_dict(orient="records")[0],
        vs_attrs_df=dt["vertices"],
        es_attrs_df=dt["edges"],
        calc_length=False,
        use_vids=True,
    )


def load_penetrating_tree(
    tree_type: Literal["DA", "AV"],
    id: Union[str, int],
    scale: float = 1,
    rotate: float = 0,
    min_diam: float = -np.inf,
    graph_name: str = ""
) -> Graph:
    """Load a penetrating vessel tree whose root locates at (0, 0, 0)."""
    path_base = f"{config.dir_pt_trees[tree_type]}/{id}"
    path_es_dict = f"{path_base}_edgesDict.pkl"
    path_vs_dict = f"{path_base}_verticesDict.pkl"

    with open(path_es_dict, 'rb') as f:
        esdf = pickle.load(f, encoding='latin1')
    with open(path_vs_dict, 'rb') as f:
        vsdf = pickle.load(f, encoding='latin1')

    adjlist = np.array(esdf["tuple"])

    g = Graph(adjlist.tolist())

    ## graph-level attributes {{{
    if graph_name:
        g['name'] = graph_name
    else:
        g['name'] = f"Penetrating tree ({tree_type}) #{id}"
    ## }}}

    ## vertex attributes {{{
    # is_connected2PV, is_connected2PA and is_connected2caps
    degree = np.array(g.degree())
    if tree_type == "DA":
        g.vs["is_connected2PA"] = np.equal(vsdf["attachmentVertex"], 1)
        g.vs["is_connected2PV"] = np.full(g.ecount(), fill_value=False)
        g.vs["is_connected2caps"] = np.logical_and(
            degree == 1, np.equal(g.vs["is_connected2PA"], False)
        )

        is_da_root_vs = np.equal(g.vs["is_connected2PA"], 1)
        # ignore other attachment vertices
        coord_tree_root = np.array(vsdf["coords"])[is_da_root_vs][0]
    elif tree_type == "AV":
        g.vs["is_connected2PA"] = np.full(g.ecount(), fill_value=False)
        g.vs["is_connected2PV"] = np.equal(vsdf["attachmentVertex"], 1)
        g.vs["is_connected2caps"] = np.logical_and(
            degree == 1, np.equal(g.vs["is_connected2PV"], False)
        )

        is_av_outflow_vs = np.isin(np.array(g.vs["is_connected2PV"]), 1)
        coord_tree_root = np.array(vsdf["coords"])[is_av_outflow_vs][0]

    # x, y and z coordinates
    vs_coords = np.array(vsdf["coords"])
    vs_coords_scaled = (vs_coords - coord_tree_root) * scale
    rotate_mat = np.array([[np.cos(rotate), -np.sin(rotate), 0.0],
                           [np.sin(rotate), np.cos(rotate) , 0.0],
                           [0.0           , 0.0            , 1.0]])
    vs_coords_rotated = (np.dot(rotate_mat, np.transpose(vs_coords_scaled))).T
    g.vs["x"], g.vs["y"], g.vs["z"] = vs_coords_rotated.T
    ## }}}

    ## edge attributes {{{
    # caculate edge lengths
    edge_lengths = np.zeros(g.ecount(), dtype=np.double)
    for i_e in range(g.ecount()):
        edge_vs = esdf["points"][i_e]
        edge_segments = (np.roll(edge_vs, shift=1, axis=0) - edge_vs)[1:]
        edge_lengths[i_e] = np.sum(np.linalg.norm(edge_segments, axis=1)) \
            * scale
    g.es["length"] = edge_lengths

    # diameter (apply min_diam)
    diameter = np.array(esdf["diameter"])
    diameter[diameter < min_diam] = min_diam
    g.es["diameter"] = diameter

    # g.es["is_stroke"] = np.full(g.ecount(), fill_value=False)
    # 0-PA; 1-PV; 2-DA; 3-AV; 4-C
    g.es["type"] = np.full(g.ecount(),
                           fill_value=2 if tree_type == "DA" else 3)
    ## }}}

    return g


