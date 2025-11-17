import numpy as np

def change_graph_attributes_casespecific_test_1(graph):
    # fix an issue, where 0 was used in Attribute vRBC, instead of None

    vRBC_pre_exp = []
    vRBC_pre_larger_10 = []
    for val in graph.es['vRBC_pre_exp']:
        if val == 0 and type(val) == int:
            vRBC_pre_exp.append(None)
            vRBC_pre_larger_10.append(0)
        else:
            vRBC_pre_exp.append(val)
            if np.abs(val)>10:
                vRBC_pre_larger_10.append(1)
            else:
                vRBC_pre_larger_10.append(0)

    vRBC_post_exp = []
    for val in graph.es['vRBC_post_exp']:
        if val == 0 and type(val) == int:
            vRBC_post_exp.append(None)
        else:
            vRBC_post_exp.append(val)

    del graph.es['vRBC_pre_exp'], graph.es['vRBC_post_exp']
    graph.es['vRBC_pre_exp'] = vRBC_pre_exp
    graph.es['vRBC_post_exp'] = vRBC_post_exp

    # Add attributes "added_manually", "vRBC_pre_larger10"
    # change graph attribute name "is_collateral"

    graph.es['is_collateral'] = graph.es['isCollateral']
    graph.es['added_manually'] = 0

    del graph.es['isCollateral']

    graph.es['vRBC_pre_larger10'] = vRBC_pre_larger_10

    return graph

def change_graph_attributes_casespecific_c57bl6_1(graph):
    # fix an issue, where 0 was used in Attribute vRBC, instead of None

    vRBC_pre_exp = []
    vRBC_pre_larger_10 = []
    for val in graph.es['vRBC_pre_exp']:
        if val == 0 and type(val) == int:
            vRBC_pre_exp.append(None)
            vRBC_pre_larger_10.append(0)
        else:
            vRBC_pre_exp.append(val)
            if np.abs(val)>10:
                vRBC_pre_larger_10.append(1)
            else:
                vRBC_pre_larger_10.append(0)

    vRBC_post_exp = []
    for val in graph.es['vRBC_post_exp']:
        if val == 0 and type(val) == int:
            vRBC_post_exp.append(None)
        else:
            vRBC_post_exp.append(val)

    # Fix the direction of two measurements
    vRBC_pre_exp = np.array(vRBC_pre_exp)
    vRBC_post_exp = np.array(vRBC_post_exp)
    index_exp = np.array(graph.es['index_exp'])
    vRBC_pre_exp[np.isin(index_exp, 23)] *= (-1)
    vRBC_pre_exp[np.isin(index_exp, 12)] *= (-1)
    vRBC_post_exp[np.isin(index_exp, 23)] *= (-1)
    vRBC_post_exp[np.isin(index_exp, 12)] *= (-1)

    del graph.es['vRBC_pre_exp'], graph.es['vRBC_post_exp']
    graph.es['vRBC_pre_exp'] = vRBC_pre_exp
    graph.es['vRBC_post_exp'] = vRBC_post_exp

    # Add attributes "added_manually", "vRBC_pre_larger10"
    # change graph attribute name "is_collateral"

    graph.es['is_collateral'] = graph.es['isCollateral']
    graph.es['added_manually'] = 0

    del graph.es['isCollateral']

    graph.es['vRBC_pre_larger10'] = vRBC_pre_larger_10

    return graph


def change_graph_attributes_casespecific_c57bl6_2(graph):

    # Add attributes "added_manually", "vRBC_pre_larger10"
    # change graph attribute name "is_collateral"

    graph.es['is_collateral'] = graph.es['isCollateral']

    graph.es['diam_post_exp'] = -1
    graph.es['vRBC_post_exp'] = None

    # assign aca in and mca in
    aca_in = np.zeros(graph.vcount(), dtype=int)
    aca_in[np.array([11,43])] = 1
    del graph.vs['ACA_in']
    graph.vs['ACA_in'] = aca_in

    mca_in = np.zeros(graph.vcount(), dtype=int)
    mca_in[np.array([0,25])] = 1
    del graph.vs['MCA_in']
    graph.vs['MCA_in'] = mca_in

    # delete connection edges. will be added again in next step
    graph.delete_vertices(np.array([50,51,52,53]))

    del graph.es['isCollateral']
    del graph.es['diam_post90_exp'], graph.es['diam_post30_exp'], graph.es['diam_post120_exp'], graph.es['diam_post60_exp'], graph.es['diam_post0_exp']
    del graph.es['vRBC_post30_larger10'], graph.es['vRBC_post60_larger10'], graph.es['vRBC_post120_larger10'], graph.es['vRBC_post90_larger10'], graph.es['vRBC_post0_larger10']
    del graph.es['vRBC_post60_exp'], graph.es['vRBC_post30_exp'], graph.es['vRBC_post90_exp'], graph.es['vRBC_post0_exp'], graph.es['vRBC_post120_exp']

    return graph


def change_graph_attributes_casespecific_balbc_1(graph):

    # fix an issue, where 0 was used in Attribute vRBC, instead of None

    vRBC_pre_exp = []
    vRBC_pre_larger_10 = []
    for val in graph.es['vRBC_pre_exp']:
        if val == 0 and type(val) == int:
            vRBC_pre_exp.append(None)
            vRBC_pre_larger_10.append(0)
        else:
            vRBC_pre_exp.append(val)
            if np.abs(val)>10:
                vRBC_pre_larger_10.append(1)
            else:
                vRBC_pre_larger_10.append(0)

    vRBC_post_exp = []
    for val in graph.es['vRBC_post_exp']:
        if val == 0 and type(val) == int:
            vRBC_post_exp.append(None)
        else:
            vRBC_post_exp.append(val)

    del graph.es['vRBC_pre_exp'], graph.es['vRBC_post_exp']
    graph.es['vRBC_pre_exp'] = vRBC_pre_exp
    graph.es['vRBC_post_exp'] = vRBC_post_exp

    # Add attributes "added_manually", "vRBC_pre_larger10"
    # change graph attribute name "is_collateral"

    graph.es['is_collateral'] = 0
    graph.es['added_manually'] = 0

    graph.es['vRBC_pre_larger10'] = vRBC_pre_larger_10

    return graph


def change_graph_attributes_casespecific_balbc_2(graph):

    # Add attributes "added_manually", "vRBC_pre_larger10"
    # change graph attribute name "is_collateral"

    graph.es['is_collateral'] = 0

    graph.es['diam_post_exp'] = -1
    graph.es['vRBC_post_exp'] = None

    mca_in = np.zeros(graph.vcount(), dtype=int)
    mca_in[np.array([0,34])] = 1
    del graph.vs['MCA_in']
    graph.vs['MCA_in'] = mca_in

    # delete connection edges. will be added again in next step
    graph.delete_vertices(np.array([82,83]))

    del graph.es['diam_post90_exp'], graph.es['diam_post30_exp'], graph.es['diam_post120_exp'], graph.es['diam_post60_exp'], graph.es['diam_post0_exp']
    del graph.es['vRBC_post30_larger10'], graph.es['vRBC_post60_larger10'], graph.es['vRBC_post120_larger10'], graph.es['vRBC_post90_larger10'], graph.es['vRBC_post0_larger10']
    del graph.es['vRBC_post60_exp'], graph.es['vRBC_post30_exp'], graph.es['vRBC_post90_exp'], graph.es['vRBC_post0_exp'], graph.es['vRBC_post120_exp']

    return graph


def add_connection2_cow_test_1(graph):

    print('No need to add circle of willis for test_1. DONE.')

    return graph


def add_connection2_cow_c57bl6_1(graph, l_stroke=50):

    print('Manually adding new vertices and edges for connection to circle of willis...')

    coords_86 = .5 * (graph.vs['coords'][84] + graph.vs['coords'][85]) + np.array([0, 800., 0])
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_86, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    coords_87 = coords_86 + np.array([-2000, 0, 0])
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_87, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    coords_88 = coords_87 + np.array([-1000, -1750, 0])
    graph.add_vertex(COW_in=1, ACA_in=0, MCA_in=0, coords=coords_88, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    graph.add_edge(84, 86, diameter=40., length=1000., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(85, 86, diameter=40., length=1000., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(86, 87, diameter=50., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(87, 88, diameter=65., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    coords_89 = coords_88 + np.array([250, -2000, 0])
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_89, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    coords_90 = coords_88 + np.array([150, -1000, 0])
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_90, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    coords_91 = coords_89 + np.array([l_stroke,0.,0.])
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_91, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    graph.add_edge(12, 91, diameter=60., length=1000. - l_stroke, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(91, 89, diameter=60., length=l_stroke, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=1, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(89, 90, diameter=92., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(90, 88, diameter=94., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    print('Manually adding new vertices and edges for connection to circle of willis. DONE.')

    return graph


def add_connection2_cow_c57bl6_2(graph, l_stroke=50.):

    print('Manually adding new vertices and edges for connection to circle of willis...')

    # 56
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([1800, -3450, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 57
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([-200, -3200, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 58
    graph.add_vertex(COW_in=1, ACA_in=0, MCA_in=0, coords=np.array([-1000, -1600, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 59
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([-800, 0, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 60
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([-250, 900, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 61
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([-150, 1000, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 62
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([400, 1150, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # ACA edges

    graph.add_edge(11, 56, diameter=40., length=1000., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(43, 56, diameter=40., length=1000., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(56, 57, diameter=50., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(57, 58, diameter=65., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    # MCA edges

    graph.add_edge(0, 62, diameter=55., length=500., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(25, 62, diameter=55., length=500., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(62, 61, diameter=60., length=500. - l_stroke, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(61, 60, diameter=60., length=l_stroke, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=1, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(60, 59, diameter=92., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(59, 58, diameter=94., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    print('Manually adding new vertices and edges for connection to circle of willis. DONE.')

    return graph


def add_connection2_cow_balbc_1(graph, l_stroke=50):

    print('Manually adding new vertices and edges for connection to circle of willis...')

    coords_78 = .5 * (graph.vs['coords'][24] + graph.vs['coords'][21]) + np.array([0, -500., 0])
    coords_79 = .5 * (graph.vs['coords'][0] + graph.vs['coords'][47]) + np.array([-100, -500., 0])
    coords_80 = graph.vs['coords'][6] + np.array([-200, -500., 0])
    coords_81 = (coords_78+coords_79)*.5 + np.array([-200, -400., 0])

    coords_82 = np.array([-1200,300,0])
    coords_77 = np.array([-2200,1500,0])

    coords_83 = graph.vs['coords'][15] + np.array([-300, -300, 0])
    coords_84 = coords_83 + np.array([-30,-30,0])
    coords_85 = np.array([-2000,2800,0])

    graph.add_vertex(COW_in=1, ACA_in=0, MCA_in=0, coords=coords_77, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_78, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_79, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_80, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_81, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_82, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_83, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_84, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=coords_85, is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    graph.add_edge(24, 78, diameter=40, length=500, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(21, 78, diameter=40, length=500, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(0, 79, diameter=40, length=500, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(47, 79, diameter=40, length=500, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(6, 80, diameter=40, length=500, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(55, 80, diameter=40, length=500, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(78, 81, diameter=45, length=500, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(79, 81, diameter=45, length=500, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(80, 81, diameter=45, length=500, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(81, 82, diameter=50, length=2250, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(77, 82, diameter=65, length=2250, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(15, 83, diameter=60, length=1000. - l_stroke, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(83, 84, diameter=60, length=l_stroke, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=1, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(84, 85, diameter=92, length=2250, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(77, 85, diameter=94, length=2250, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)


    print('Manually adding new vertices and edges for connection to circle of willis. DONE.')

    return graph


def add_connection2_cow_balbc_2(graph, l_stroke=50.):

    print('Manually adding new vertices and edges for connection to circle of willis...')

    # 84
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([2000, -2400, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 85
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([400, -2000, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 86
    graph.add_vertex(COW_in=1, ACA_in=0, MCA_in=0, coords=np.array([-800, -800, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 87
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([-700, 300, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 88
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([-400, 1100, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 89
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([-300, 1200, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # 90
    graph.add_vertex(COW_in=0, ACA_in=0, MCA_in=0, coords=np.array([0, 1400, 0.]), is_connected_2caps=0, is_AV_root=0,
                     is_DA_startingPt=0, is_DA_startingPt_added_manually = 0)

    # ACA edges

    graph.add_edge(12, 84, diameter=50., length=1000., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(84, 85, diameter=50., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)
    graph.add_edge(85, 86, diameter=65., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    # MCA edges

    graph.add_edge(0, 90, diameter=60., length=700., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(34, 90, diameter=60., length=700., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(90, 89, diameter=60., length=300. - l_stroke, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(89, 88, diameter=60., length=l_stroke, type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=1, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(88, 87, diameter=92., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    graph.add_edge(87, 86, diameter=94., length=2250., type=-1, diam_post_exp=-1, diam_pre_exp=-1, index_exp=-1,
                   is_collateral=0, vRBC_post_exp=None, vRBC_pre_exp=None, is_stroke=0, added_manually = 1, vRBC_pre_larger10 = 0)

    print('Manually adding new vertices and edges for connection to circle of willis. DONE.')

    return graph
