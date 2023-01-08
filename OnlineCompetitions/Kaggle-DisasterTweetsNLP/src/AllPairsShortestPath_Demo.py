import numpy as np
import pandas as pd
from scipy.sparse.csgraph import floyd_warshall

data = [
    [0, 1, 7],
    [2, 3, 5],
    [4, 6],
    [5, 7]
]

flat_data = [item for sublist in data for item in sublist]
unique_person, freq = np.unique(np.array(flat_data), return_counts=True)
nunique_person = len(unique_person)

adj_mat = np.zeros((nunique_person, nunique_person))
for l in data:
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            adj_mat[l[i], l[j]] = 1
            adj_mat[l[j], l[i]] = 1


APSP, pred = floyd_warshall(csgraph=adj_mat, directed=False, return_predecessors=True)
APSP = APSP-1

pd.DataFrame(APSP).to_csv('/home/vss/Documents/Work/Misc/AllPairsShortestPath-Graph/ShortestPath.csv')
pd.DataFrame(pred).to_csv('/home/vss/Documents/Work/Misc/AllPairsShortestPath-Graph/Pred-ShortestPath.csv')