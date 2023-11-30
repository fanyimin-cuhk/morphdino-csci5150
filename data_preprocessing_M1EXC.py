from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import networkx as nx
from allensdk.core.cell_types_cache import CellTypesCache
import numpy as np
from codebase.ssl_neuron.data.data_utils import connect_graph, remove_axon, rotate_cell
from codebase.ssl_neuron.utils import neighbors_to_adjacency, plot_neuron
import os
import numpy as np

import pandas as pd


root_dir = '/mnt/data/aim/liyaxuan/projects/project2_pre/M1_EXC/'
cell_allids = np.load(root_dir + 'all_ids.npy')
cell_alllabels = np.load(root_dir + 'all_labels.npy')
n = len(cell_allids)

train_ids = []
train_labels = []

val_ids = []
val_labels = []

for i in range(n):
    if np.random.rand() < 0.1:
        val_ids.append(cell_allids[i])
        val_labels.append(cell_alllabels[i])
    else:
        train_ids.append(cell_allids[i])
        train_labels.append(cell_alllabels[i])

np.save(root_dir + 'train_ids.npy', train_ids)
np.save(root_dir + 'train_labels.npy', train_labels)

np.save(root_dir + 'val_ids.npy', val_ids)
np.save(root_dir + 'val_labels.npy', val_labels)
