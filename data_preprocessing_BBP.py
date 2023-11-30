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

root_dir = '/mnt/data/aim/liyaxuan/projects/project2_pre/BBP/'
clss = ['L1_DAC', 'L1_DLAC', 'L1_HAC', 'L1_NGC-DA', 'L1_NGC-SA', 'L1_SLAC', 'L23_BP', 'L23_BTC', 'L23_ChC', 'L23_DBC', 'L23_LBC', 'L23_MC', 'L23_NBC', 'L23_NGC', 'L23_PC', 'L23_SBC', 'L4_BP', 'L4_BTC', 'L4_ChC', 'L4_DBC', 'L4_LBC', 'L4_MC', 'L4_NBC', 'L4_NGC', 'L4_SBC', 'L4_PC', 'L4_SP', 'L4_SS', 'L5_BP', 'L5_BTC', 'L5_ChC', 'L5_DBC', 'L5_LBC', 'L5_MC', 'L5_NBC', 'L5_NGC', 'L5_SBC', 'L5_STPC', 'L5_TTPC1', 'L5_TTPC2', 'L5_UTPC', 'L6_BPZ', 'L6_BPC', 'L6_BTC', 'L6_ChC', 'L6_DBC', 'L6_IPC', 'L6_LBC', 'L6_MC', 'L6_NBC', 'L6_NGC', 'L6_SBC', 'L6_TPC_L1', 'L6_TPC_L4', 'L6_UTPC']

all_ids = np.load(root_dir + 'all_ids.npy')
tmp_labels = np.load(root_dir + 'all_labels.npy')
all_labels = []
for i in tmp_labels:
    all_labels.append(i)
all_labels.append(-1)

n = len(all_ids)
train_max = int(n * 0.9)
num_train = 0
num_val = 0

val_data = []
train_data = []

for i in range(n):
    now = (all_ids[i], all_labels[i])
    if ((i == 0) or (all_labels[i] != all_labels[i-1]) or (num_train > train_max or np.random.rand() < 0.1)) and (all_labels[i] == all_labels[i+1]):
        val_data.append(now)
    else:
        train_data.append(now)
        num_train += 1

np.save(root_dir + 'train_data.npy', train_data)
np.save(root_dir + 'val_data.npy', val_data)

root_dir = '/mnt/data/aim/liyaxuan/projects/project2_pre/BBP/'
val_data = np.load(root_dir + 'val_data.npy')
train_data = np.load(root_dir + 'train_data.npy')
clss = ['L1_DAC', 'L1_DLAC', 'L1_HAC', 'L1_NGC-DA', 'L1_NGC-SA', 'L1_SLAC', 'L23_BP', 'L23_BTC', 'L23_ChC', 'L23_DBC', 'L23_LBC', 'L23_MC', 'L23_NBC', 'L23_NGC', 'L23_PC', 'L23_SBC', 'L4_BP', 'L4_BTC', 'L4_ChC', 'L4_DBC', 'L4_LBC', 'L4_MC', 'L4_NBC', 'L4_NGC', 'L4_SBC', 'L4_PC', 'L4_SP', 'L4_SS', 'L5_BP', 'L5_BTC', 'L5_ChC', 'L5_DBC', 'L5_LBC', 'L5_MC', 'L5_NBC', 'L5_NGC', 'L5_SBC', 'L5_STPC', 'L5_TTPC1', 'L5_TTPC2', 'L5_UTPC', 'L6_BPZ', 'L6_BPC', 'L6_BTC', 'L6_ChC', 'L6_DBC', 'L6_IPC', 'L6_LBC', 'L6_MC', 'L6_NBC', 'L6_NGC', 'L6_SBC', 'L6_TPC_L1', 'L6_TPC_L4', 'L6_UTPC']
clss_num = len(clss)
clss_tot_val = [0 for i in range(clss_num)] 
clss_tot_train = [0 for i in range(clss_num)] 
for i in val_data:
    clss_tot_val[i[1]] += 1
for i in train_data:
    clss_tot_train[i[1]] += 1
        
for i in range(clss_num):
    print('{}: train({}), val({})'.format(clss[i], clss_tot_train[i], clss_tot_val[i]))

print('tot_num: train({}), val({})'.format(len(train_data), len(val_data)))