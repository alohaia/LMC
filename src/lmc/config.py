"""
- attributes:
  - edge:
    - is_added_manually: include COW arteries and edges connecting new DA roots
"""

import numpy as np

graph_attrs = {
    "graph": ["name", "gen_date"],
    "vertex": ["name", "x", "y", "z", "is_ACA_in", "is_MCA_in", "COW_in",
               "is_AV_root", "is_DA_root","is_DA_root_added_manually",
               "is_connected_2caps", "type"],
    "edge": ["is_added_manually", "diam_post_exp", "diam_pre_exp", "diameter",
             "index_exp", "is_collateral", "is_stroke", "length",
             "vRBC_post_exp", "vRBC_pre_exp", "is_vRBC_pre_larger10", "type"]
}

palette = {
    "DA root": "#FF287F",
    "DA root added manually": "#FF9855",
    "AV root" : "#0943AA",
    "MCA inflow": "#7EA5E0",
    "ACA inflow": "#97E07E",
}

dir_pt_trees = {
    "DA":
        "Research_Data/Scripts_Network_Generation/penetrating_trees/arteries/",
    "AV": "Research_Data/Scripts_Network_Generation/penetrating_trees/veins/"
}

pttree_candidates = {
    "DA": np.array([
        0, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 26, 27,
        28, 29, 32, 33, 35, 36, 37, 39, 42, 44, 45, 46, 47, 48, 49, 50, 52, 54,
        55, 56
    ], dtype=np.int_),
    "AV": np.array([
        0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 33, 34, 35, 36, 39, 42, 43, 44, 45, 47, 50, 51, 52, 54, 57, 58, 59,
        66, 67, 69, 71, 73, 74, 75, 76, 79, 80, 81, 82, 86, 88, 90, 91, 92, 93,
        97, 98, 99
    ], dtype=np.int_)
}
