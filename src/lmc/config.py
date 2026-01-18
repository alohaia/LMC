'''
- attributes:
  - edge:
    - is_added_manually: This attribute is used to distinguish manually PA
      edges, including CoW arteries and edges connecting new DA roots.
'''

import numpy as np

random_seed = 666

mice = [
    '05_PBS', '12_PBS', '13_PBS',
    '08_norEVs', '09_norEVs', '02_norEVs',
    '01_hypEVs', '07_hypEVs', '11_hypEVs'
]
# mice = ['08_norEVs']
diameter_attrs = ['diameter', 'diameter_mcao0h', 'diameter_mcao1h']
is_cow_added = True

graph_attrs = {
    'before_adding_CoW': {
        'graph': ['name', 'gen_date'],
        'vertex':  ['name', 'x', 'y', 'z', 'is_DA_root', 'is_MCA_in', 'is_ACA_in',
                    'is_added_manually', 'type'],
        'edge': ['diameter', 'diameter_mcao0h', 'diameter_mcao1h', 'is_collateral',
                 'length', 'type', 'is_added_manually']
    },
    'after_adding_CoW': {
        'graph': ['name', 'gen_date'],
        'vertex':  ['name', 'x', 'y', 'z', 'is_DA_root', 'is_MCA_in', 'is_ACA_in',
                    'is_added_manually', 'type', 'is_CoW_in'],
        'edge': ['diameter', 'diameter_mcao0h', 'diameter_mcao1h', 'is_collateral',
                 'type', 'length', 'is_added_manually',
                 'is_stoke', 'type', 'is_CoW_in']
    }
}

stoke_length = 50
es_length_map = {
    -11: 2250,
    -12: 2250,
    -13: 1000,
    -15: 500,

    -21: 2250,
    -22: 2250,
    -23: 1000 - stoke_length,
    -24: stoke_length,
    -25: 500 - stoke_length,
    -26: 500,
}
es_diameter_map = {
    -11: 65,
    -12: 50,
    -13: 40,
    -15: 40,

    -21: 94,
    -22: 92,
    -23: 60,
    -24: 60,
    -25: 55,
    -26: 55
}

palette = {
    'DA root': '#FF287F',
    'DA root added manually': '#FF9855',
    'AV root' : '#0943AA',
    'MCA inflow': '#7EA5E0',
    'ACA inflow': '#97E07E',
}

dir_pt_trees = {
    'DA':
        'Research_Data/Scripts_Network_Generation/penetrating_trees/arteries/',
    'AV': 'Research_Data/Scripts_Network_Generation/penetrating_trees/veins/'
}

pttree_candidates = {
    'DA': np.array([
        0, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 26, 27,
        28, 29, 32, 33, 35, 36, 37, 39, 42, 44, 45, 46, 47, 48, 49, 50, 52, 54,
        55, 56
    ], dtype=np.int_),
    'AV': np.array([
        0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 33, 34, 35, 36, 39, 42, 43, 44, 45, 47, 50, 51, 52, 54, 57, 58, 59,
        66, 67, 69, 71, 73, 74, 75, 76, 79, 80, 81, 82, 86, 88, 90, 91, 92, 93,
        97, 98, 99
    ], dtype=np.int_)
}
