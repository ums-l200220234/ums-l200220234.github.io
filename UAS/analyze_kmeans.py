from itertools import islice
import numpy as np


def top_data_points(num_clusters, clusters, mtx, columns):
    top = []
    for i in range(num_clusters):
        rows_in_cluster = np.where(clusters == i)[0]
        word_freqs = mtx[rows_in_cluster].sum(axis=0).A[0]
        ordered_freqs = np.argsort(word_freqs)
        top_words = [(columns[idx], int(word_freqs[idx]))
                     for idx in islice(reversed(ordered_freqs), 3)]
        top.append(top_words)
    return top

from metaflow import Flow
run = Flow('KMeansFlow').latest_run

k = run.data.top[4]

print(k)