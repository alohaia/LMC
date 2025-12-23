"""
- attributes:
  - edge:
    - is_added_manually: include COW arteries and edges connecting new DA roots
"""

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

dir_penetrating_trees = "Research_Data/Scripts_Network_Generation/penetrating_trees/"

