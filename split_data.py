import torch.nn as nn
import torch
import numpy as np

root_dir = '/mnt/data/aim/liyaxuan/projects/project2/M1_EXC/'
split_p = 0.4

cell_allids = np.load(root_dir + 'all_ids.npy')
cell_alllabels = np.load(root_dir + 'all_labels.npy')

n = len(cell_allids)

train_ids = []
train_labels = []

val_ids = []
val_labels = []

for i in range(n):
    if np.random.rand() < split_p:
        val_ids.append(cell_allids[i])
        val_labels.append(cell_alllabels[i])
    else:
        train_ids.append(cell_allids[i])
        train_labels.append(cell_alllabels[i])

np.save(root_dir + 'train_ids.npy', train_ids)
np.save(root_dir + 'train_labels.npy', train_labels)

np.save(root_dir + 'val_ids.npy', val_ids)
np.save(root_dir + 'val_labels.npy', val_labels)
