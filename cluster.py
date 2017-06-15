from segment import *
from sdtw import *
from config import *
from functools import reduce
from itertools import accumulate

import numpy as np
import scipy.sparse as sp
import scipy
import matplotlib.pyplot as plt

# segments are a list of short mfcc feature - time sequences
def build_matrix(segments):
    n_segments = len(segments)
    segment_lengths = list(map(lambda s: s.shape[1], segments))
    accumulate_lengths = [0] + list(accumulate(segment_lengths))
    total_length = accumulate_lengths[-1]

    # print(total_length, accumulate_lengths)
    
    # sparse matrix representation of the graph
    row, col, data = [], [], []

    similarity_scores = [np.zeros((seg.shape[1],)) for seg in segments]
    for i in range(n_segments):
        print('I: fragment %d being matched against others' % i)
        for j in range(i+1, n_segments):
            paths = compare_signal(segments[i], segments[j])
            for path, average_distortion in paths.values():
                for coord in path:
                    if average_distortion < THETA:
                        row += [convert_to_global_index(i, coord[0], accumulate_lengths)]
                        col += [convert_to_global_index(j, coord[1], accumulate_lengths)]
                        data += [similarity_score(average_distortion)]

    similarity_coo = sp.coo_matrix((data, (row, col)), shape=(total_length, total_length))
    sim = similarity_coo.toarray()
    sim += np.transpose(sim)
    # plt.matshow(sim[0:1000, 0:1000])
    # plt.show()
    print('I: matrix built')
    return similarity_coo, accumulate_lengths

def build_graph(similarity_coo, accumulate_lengths):
    # trying to implement eq. 10
    # urrr, there is a subtle difference here.
    similarity = similarity_coo.toarray()
    sum_over_P = np.sum(similarity_coo.toarray() + np.transpose(similarity_coo.toarray()), axis=1)

    print(sum_over_P.shape)
    # plt.plot(sum_over_P)
    # plt.show()

    # divide again
    scores = [
        sum_over_P[i:j] for i,j in zip(accumulate_lengths[:-1], accumulate_lengths[1:]) 
    ]

    # instead of triangular averaing, we use gaussian for simplicity
    smoothed_similarity = [
        scipy.ndimage.filters.gaussian_filter(score, sigma=WINDOW_WIDTH/3, mode='nearest') 
        for score in scores
    ]

    local_extremas = [
        scipy.signal.argrelextrema(sim, np.greater, order=1, mode='clip')
        for sim in smoothed_similarity
    ]

    # plt.plot(np.array(reduce(lambda l1, l2 : np.concatenate([l1, l2]), smoothed_similarity)))
    # plt.show()

    nodes_global_index = reduce(
        lambda l1, l2 : l1 + l2, 
        [ [convert_to_global_index(i, j, accumulate_lengths) for j in jl]
            for i, jl in enumerate(local_extremas)
        ] 
    )

    n_nodes = reduce(lambda x, y: x+y, map(len, local_extremas))
    edge_set = set()
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if similarity[i, j] > 0:
                edge_set.add((i, j, similarity[i, j]))

    return n_nodes, edge_set

    # build edges

# get a global index for j-th feature of fragment i
def convert_to_global_index(i, j, segments_acc):
    return segments_acc[i] + j

# implements eq. 9
def similarity_score(average_distortion):
    return (THETA - average_distortion) / THETA

# E is represented by a set of 3-tuple (v1, v2, weight), 
# v1 and v2 are integers in range(0, n_nodes) 
# should return a set of connect-component as described
# in sect IV. B.

# the use of set: 
# you can access all the elements in an unordered set by:
# for e in E:
#     func(e)
# you may also convert E to another data-structure first 
# if you need to
# def cluster(n_nodes, E):
#     pass

def cluster(n_node, E):
    sum_weight=sum([e[2] for e in E ])
    newman_id = np.array(np.int32, [i for range(n_node)])
    newman_e = np.array((n_node, n_node))
    newman_a = np.zeros((n_node,))
    for e in E:
        newman_e[e[0]][e[1]] = e[2] / sum_weight
        newman_e[e[1]][e[0]] = e[2] / sum_weight
        newman_a[e[0]] = newman_a[e[0]]+newman_e[e[0]][e[1]]
        newman_a[e[1]] = newman_a[e[1]]+newman_e[e[0]][e[1]]
    modularity_Q=0
    for i in range(n_node):
        Q = Q + newman_e[i][i] - newman_a[i]*newman_a[i]
    for i in range(n_node):
        maxdelta=0
        u=-1
        v=-1
        for e in E:
            if (newman_id[e[0]]=newman_id[e[1]]):
                continue
            elif 2*newman_e[newman_id[e[0]]][newman_id[e[1]]]-2*newman_a[newman_id[e[0]]]*newman_a[newman_id[e[1]]]>maxdelta:
                maxdelta=2*newman_e[newman_id[e[0]]][newman_id[e[1]]]-2*newman_a[newman_id[e[0]]]*newman_a[newman_id[e[1]]]
                u=newman_id[e[0]]
                v=newman_id[e[1]]
        if Q+maxdelta<bound:
            break
        for j in range(0,n_node):
            if newman_id[i]=v:
                newman_id[i]=u
            elif newman_id[i]=i:
                newman_e[u][i] = newman_e[u][i] + newman_e[v][i]
                newman_e[i][u] = newman_e[u][i]
        newman_a[u] = newman_a[u] + newman_a[v]
    return newman_id

        

if __name__ == '__main__':
    V, E = build_graph(*build_matrix(load_feature()))

    # examples of how cluster() will be called.
    clusters = cluster(V, E)

