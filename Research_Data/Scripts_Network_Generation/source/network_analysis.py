import numpy as np

def get_eids_vs_roots(graph, type=2):

    if type == 2:
        is_root = np.array(graph.vs['is_DA_startingPt']) > 0
    elif type == 3:
        is_root = np.array(graph.vs['is_AV_root']) > 0
    else:
        is_root = np.array([False]*graph.vcount())

    vids_roots = np.arange(graph.vcount())[is_root]
    nr_of_roots = np.size(vids_roots)
    eids_roots = -np.ones(nr_of_roots, dtype=np.int)

    for vid_root, i in zip(vids_roots, range(nr_of_roots)):

        vid_nbs = np.array(graph.neighbors(vid_root))  # find neighbours of current root

        for vid_nb in vid_nbs:
            eid_nb = graph.get_eid(vid_root, vid_nb)  # find eid of edge adjacent to head node
            if graph.es['type'][eid_nb] == type:
                if eids_roots[i] > -1:
                    print "Error... More than one daughter edge to root node"
                else:
                    eids_roots[i] = eid_nb
            else:
                continue

    return vids_roots, eids_roots