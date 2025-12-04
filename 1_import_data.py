#!/usr/bin/env python

import sys
from src import network

import numpy as np
import pandas as pd

import igraph

# Load data and create a Graph object.
# The data contains the manually added COW.
graph_SA = network.create("data/test 1/vertices.csv", "data/test 1/edges.csv")
igraph.plot(
    graph_SA,
    layout=[arr[:2] for arr in graph_SA.vs["coords"]],
    vertex_size=4,
    vertex_color="lightblue",
    edge_color="gray",
    bbox=(600, 600),
    margin=30
)

# check the network
print(graph_SA.summary())
if not network.check_attrs(graph_SA):
  sys.exit(-1)


# refine DA roots
xy_vs = np.array(graph_SA.vs["coords"])[:, 0:2]
xy_DA_roots = xy_vs[graph_SA.vs["is_DA_startingPt"], ]
xy_ACA_MCA_in = \
    xy_vs[np.array(graph_SA.vs["ACA_in"]) | np.array(graph_SA.vs["MCA_in"]), ]

xy_coords_input = np.concatenate((xy_DA_roots, xy_ACA_MCA_in), axis=0)

# xy_DA_roots =
xy_DA_roots_new_points, polygon_vs_xy = network.refine_DA_density(
    xy_DA_roots,
    ghp_boundary_offset=800, ghp_simplify_tolerance=10,
    # new DA attrs
    nr_new_DAs_max=200,
    sample_min_DA_distance=200.,
    newDA_prior_target_density=11.5,
    max_nr_of_tries=50000,
    save_path="nw_output/DA_AV_locations/",
    save_refinement_steps=False,
    save_init_final_distribution=True,
    return_voronoi_polygon_vs_coords=True
)
