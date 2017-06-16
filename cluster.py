from segment import *
from sdtw import *
from config import *
from functools import reduce
from itertools import accumulate
import pickle as pkl

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
    
    # sparse matrix representation of the graph
    sim = np.zeros((total_length, total_length))
    # right_inclusive
    path_intervals = set()
    for i in range(n_segments):
        print('I: fragment %d being matched against others' % i)
        for j in range(i+1, n_segments):
            paths = compare_signal(segments[i], segments[j])
            for path, average_distortion in paths.values():
                if average_distortion < THETA:
                    for coord in path:
                        row = convert_to_global_index(i, coord[0], accumulate_lengths)
                        col = convert_to_global_index(j, coord[1], accumulate_lengths)
                        # data += [similarity_score(average_distortion)]
                        sim[row, col] = sim[col, row] = similarity_score(average_distortion)
                    i0, j0 = path[0]
                    i1, j1 = path[-1]
                    gi0, gi1 = (convert_to_global_index(i, g, accumulate_lengths) for g in [i0, i1])
                    gj0, gj1 = (convert_to_global_index(j, g, accumulate_lengths) for g in [j0, j1])
                    assert gi0 < gi1 < gj0 < gj1
                    path_intervals.add((gi0, gi1, gj0, gj1, average_distortion))
    sim += np.transpose(sim)
    mm=1000
    plt.matshow(sim[accumulate_lengths[5]:accumulate_lengths[10], accumulate_lengths[5]:accumulate_lengths[10]])
    plt.show()
    pkl_dump('build_graph_args', (sim, accumulate_lengths, path_intervals))
    print('I: matrix built, %d paths found, variables dumped' % len(path_intervals))
    return sim, accumulate_lengths, path_intervals

def build_edge_weights(N, path_intervals):
    w = np.zeros((N, N))
    for p in path_intervals:
        # w[p[0]:p[1], p[2]:p[3]]  = np.maximum(similarity_score(p[4]), w[p[0]:p[1], p[2]:p[3]])
        w[p[0]:p[1], p[2]:p[3]]  += similarity_score(p[4])
    # print(sum(w))
    w += np.transpose(w)
    mm=2000
    # plt.matshow(w[:mm,:mm])
    # plt.show()
    return w

# compute the average start and end, IV. C in the paper
# nodes and interval limits are global indices.
def compute_node_interval(nodes, path_intervals, acc):
    intervals = [[] for i in range(len(nodes))]
    print(nodes) 
    for i, n in enumerate(nodes):
        for p in path_intervals:
            if p[0] <= n <= p[1]:
                intervals[i] += [(p[0], p[1])]
            elif p[2] <= n <= p[3]:
                intervals[i] += [(p[2], p[3])]
    print(intervals)
    float_limits = map(
        lambda limits: (np.mean([limit[0] for limit in limits]), np.mean([limit[1] for limit in limits])),
        intervals
    )
    integer_limits = list(map(
        lambda limits : (int(limits[0]), int(limits[1]) + 1),
        float_limits
    ))

    print(integer_limits)

    local_integer_limits = [(0, 0) for i in range(len(nodes))]
    # convert to local indices
    print(len(nodes), len(integer_limits), len(acc))
    filenames = []
    for i, limits in enumerate(integer_limits):
        l, r = convert_to_local_index(limits[0], acc), convert_to_local_index(limits[1], acc)
        assert l[0] == r[0]
        local_integer_limits[i] = (l[1], r[1])
        filenames += [write_node_wav(l[0], l[1], r[1])]

    print(local_integer_limits)
    # convert to time 
    local_time = [(l[0]/50, l[1]/50) for l in local_integer_limits]
    print(local_time)
    return filenames

import os
import shutil
def copy_by_group(id, filenames, profile_name):
    d = '%s%s_clusters' % (OUTPUT_PATH, profile_name)
    print(d, 'group')
    if not os.path.isdir(d):
        os.mkdir(d)
    for i in id:
        if not os.path.isdir('%s/%d' % (d, i)):
            os.mkdir('%s/%d' % (d, i))
    for i in range(len(id)):
        shutil.copy(filenames[i], '%s/%d'%(d, id[i]))

def build_graph(similarity, accumulate_lengths, path_intervals):
    ew = build_edge_weights(accumulate_lengths[-1], path_intervals)
    # trying to implement eq. 10
    # urrr, there is a subtle difference here.
    sum_over_P = np.sum(similarity, axis=1)

    print(sum_over_P.shape)
    plt.plot(sum_over_P)
    plt.show()
    # divide again
    scores = [
        sum_over_P[i:j] for i,j in zip(accumulate_lengths[:-1], accumulate_lengths[1:]) 
    ]

    # instead of triangular averaing, we use gaussian for simplicity
    smoothed_similarity = [
        scipy.ndimage.filters.gaussian_filter(score, sigma=WINDOW_WIDTH/3, mode='nearest') 
        for score in scores
    ]

    local_extrema = [
        scipy.signal.argrelextrema(sim, np.greater, order=1, mode='clip')[0]
        for sim in smoothed_similarity
    ]

    plt.plot(np.array(reduce(lambda l1, l2 : np.concatenate([l1, l2]), smoothed_similarity)))

    nodes_global_index = reduce(
        lambda l1, l2 : l1 + l2, 
        [ [convert_to_global_index(i, j, accumulate_lengths) for j in jl]
            for i, jl in enumerate(local_extrema)
        ] 
    )

    filenames = compute_node_interval(nodes_global_index, path_intervals, accumulate_lengths)

    n_nodes = reduce(lambda x, y: x+y, map(len, local_extrema))
    edge_set = set()
    
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            e0, e1 = nodes_global_index[i], nodes_global_index[j]
            if ew[e0, e1] > 0:
                edge_set.add((i, j, ew[e0, e1]))
    pkl_dump('cluster_args', (n_nodes, edge_set, filenames))
    print('I: graph build #node: %d, #edge:%d, vars dumped' % (n_nodes, len(edge_set)))
    return n_nodes, edge_set, filenames

# get a global index for j-th feature of fragment i
def convert_to_global_index(i, j, segments_acc):
    return segments_acc[i] + j

def convert_to_local_index(i, segment_acc):
    for seg_id, l in enumerate(segment_acc):
        if l > i:
            return seg_id - 1, i - segment_acc[seg_id-1]

def write_node_wav(i, start, end):
    S, sr = rosa.core.load('%s%d.wav' % (SEGMENTED_PATH, i))
    filename = '%s%02d_%.2f_%.2f.wav' % (NODE_PATH, i, start/50, end/50)
    write_wav(filename, S[start*450:end*450], sr)
    return filename

def local_time_align(local_extrema):
    for i, x in enumerate([[d/50 for d in e] for e in local_extrema]):
        print('%d.wav' % i, x)

def pkl_dump(name, var):
    with open(OUTPUT_PATH + name, 'wb') as f:
        pkl.dump(var, f)

def pkl_load(name):
    with open(OUTPUT_PATH + name, 'rb') as f:
        ans = pkl.load(f)
    return ans

# implements eq. 9
def similarity_score(average_distortion):
    return (THETA - average_distortion) / THETA

# E is represented by a set of 3-tuple (v1, v2, weight), 
# v1 and v2 are integers in range(0, n_nodes) 
# should return a set of connect-component as described
# in sect IV. B.
  
def cluster(n_node, E, filenames=None):
    id = np.array([i for i in range(n_node)])
    # assuming e normalized
    N_e = np.zeros((n_node, n_node))
    Q = 0
    for e in E:
        N_e[e[0], e[1]] = 0.5 * e[2]
        N_e[e[1], e[0]] = 0.5 * e[2]
    N_e = N_e / np.sum(N_e)
    N_a = np.sum(N_e, axis=1)
    # initial computing of Q
    Q = np.sum([
        N_e[i,i] - N_a[i]**2 for i in range(n_node)
    ])
    Q_peak = Q
    n_clusters = n_node
    for _ in range(n_node):
        if filenames != None and _ % 5 == 0:
            copy_by_group(id, filenames, str(_))
        max_delta = -INFINITY
        max_e = None
        del_e = []
        u, v = -1, -1
        for e in E:
            if id[e[0]] != id[e[1]]:
                delta = 2 * (N_e[id[e[0]], id[e[1]]] - N_a[id[e[0]]] * N_a[id[e[1]]])
                if delta > max_delta:
                    max_delta = delta
                    max_e = e
            else:
                del_e += [e]
        if max_e == None:
            print('edge exhausted')
            break
        E.remove(max_e)
        for e in del_e:
            E.remove(e)
        u, v, w = max_e
        Q = Q + max_delta
        Q_peak = max(Q, Q_peak)
        if Q < 0.9 * Q_peak and Q > 0:
            break
        if n_clusters < 20:
            break
        n_clusters -= 1
        merged_id = min(id[u], id[v])
        unused_id = max(id[u], id[v])
        N_a[merged_id] = N_a[id[u]] + N_a[id[v]]
        for j in range(n_node):
            if id[j] == unused_id:
                id[j] = merged_id
            elif id[j] == j: # is representative of its group
                N_e[merged_id, j] += N_e[unused_id, j]
                N_e[j, merged_id] += N_e[unused_id, j]

        for j in range(n_node):
            if id[j] != j:
                N_e[j,:] = 0
                N_e[:,j] = 0
                N_a[j] = 0
        print("Q: ", Q, '#clusters: ', n_clusters)
    return id, n_clusters

if __name__ == '__main__':
    # _ = build_matrix(load_feature())
    _ = build_graph(*pkl_load('build_graph_args'))
    # id, n_clusters = cluster(*_)
    # print(id)

