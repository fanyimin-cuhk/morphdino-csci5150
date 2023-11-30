from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import networkx as nx
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.core import swc
import numpy as np
from codebase.ssl_neuron.data.data_utils import connect_graph, remove_axon, rotate_cell
from codebase.ssl_neuron.utils import neighbors_to_adjacency, plot_neuron
import os
import numpy as np

import pandas as pd

root_dir = '/mnt/data/aim/liyaxuan/projects/project2_pre/M1_EXC/'
clss = {'untufted':0, 'tufted':1, 'other':2}
df = pd.read_csv(root_dir + 'meta_data_m_type_label.csv')
n = len(df)
cell_id = 0

cell_allids = []
cell_alllabels = []

ctc = CellTypesCache(manifest_file='cell_types/manifest.json')

for i in range(n):
    swc_path = root_dir + 'neurons/' + df['sample name'][i] + '.swc'
    label = clss[df['m-type2'][i]]

    try:
        morphology = swc.read_swc(swc_path) 
    except:
        continue

    # Get soma coordinates.
    soma = morphology.soma
    
    if soma == None:
        continue
    
    soma_pos = np.array([soma['x'], soma['y'], soma['z']])
    soma_id = soma['id']

    # Process graph.
    neighbors = {}
    idx2node = {}
    hav_err = 0
    for i, item in enumerate(morphology.compartment_list):
        # Get node features.
        sec_type = [0, 0, 0, 0, 0, 0, 0]

        if item['type'] > 7:
            hav_err = 1
            break

        sec_type[item['type'] - 1] = 1
        feat = tuple([item['x'], item['y'], item['z'], item['radius']]) + tuple(sec_type)
        idx2node[i] = feat
        
        # Get neighbors.
        neighbors[i] = set(item['children'])
        if item['parent'] != -1:
            neighbors[i].add(item['parent'])

    if hav_err == 1:
        continue

    features = np.array(list(idx2node.values()))
    
    assert ~np.any(np.isnan(features))
    
    # Normalize soma position to origin.
    norm_features = features.copy()
    norm_features[:, :3] = norm_features[:, :3] - soma_pos
    
    # Test if graph is connected.
    adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))
    G = nx.Graph(adj_matrix)
    if nx.number_connected_components(G) > 1:
        adj_matrix, neighbors = connect_graph(adj_matrix, neighbors, features)
        
    assert len(neighbors) == len(adj_matrix)
    # Remove axons.
    neighbors, norm_features, soma_id = remove_axon(neighbors, norm_features, int(soma_id))
    
    from codebase.ssl_neuron.utils import compute_node_distances
    distances = compute_node_distances(soma_id, neighbors)
    keys = [ key for key in neighbors]
    for i in range(len(keys)):
        if i not in distances.keys():
            hav_err = 1
            break
    
    if hav_err == 1:
        continue
    
    assert len(neighbors) == len(norm_features)
    assert ~np.any(np.isnan(norm_features))

    cell_id += 1
    path = Path(root_dir + 'skeletons/', str(cell_id))
    path.mkdir(parents=True, exist_ok=True)

    np.save(Path(path, 'features'), norm_features)
    with open(Path(path, 'neighbors.pkl'), 'wb') as f:
        pickle.dump(dict(neighbors), f, pickle.HIGHEST_PROTOCOL)
    
    cell_allids.append(cell_id)
    cell_alllabels.append(label)

np.save(root_dir + 'all_ids.npy', cell_allids)
np.save(root_dir + 'all_labels.npy', cell_alllabels)
