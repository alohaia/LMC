#!/usr/bin/env python

from pathlib import Path
import sys

from lmc import nw, viz

import numpy as np
import pandas as pd

from igraph import Graph

# Load data and create a Graph object.
# The data contains the manually added COW.
graph_SA = nw.create(Path("data/test 1/vertices.csv"),
                     Path("data/test 1/edges.csv"))

# check the network
print(graph_SA.summary())
if not nw.check_attrs(graph_SA):
    sys.exit(-1)

# refine DA roots
xys_new_DA_roots, xyss_polygon_vs = nw.refine_DA_density(
    graph_SA,
    ghp_boundary_offset=800,
    ghp_simplify_tolerance=10,
    nr_new_DAs_max=200,
    min_DA_DA_distance=200.,
    newDA_prior_target_density=11.5,
    nr_tries_max=10000, # E = density * area_netwrk
    save_path="output/test_1/",
    save_refinement_steps=True,
    save_init_final_distribution=True,
    return_voronoi_polygon_vs_coords=True
)

# xys_AVs = add_AVs(
#     xy_DA_all,
#     xyss_polygon_vs,
#     3,
#     min_dist_2_DA = 100,
#     min_dist_2_AV = 120,
#     max_nr_of_tries=10000
# )
#
# visualize.visualize_DA_AV_roots_with_polygons(
#     graph_SA, xys_new_DA_roots,
#     # xy_DA_all, # xy_DA_roots + xy_DA_roots_new_points
#     xy_AVs,
#     polygon_vs_xy,
#     show_MCA_ACA_root=False,
#     title=title_plot,
#     filepath="nw_output/DA_AV_locations/DA_AV_map.png"
# )
