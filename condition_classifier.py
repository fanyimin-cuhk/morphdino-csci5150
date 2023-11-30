import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn

from dinov2.data.datasets.utils import neighbors_to_adjacency, subsample_graph, rotate_graph, jitter_node_pos, translate_soma_pos, get_leaf_branch_nodes, compute_node_distances, drop_random_branch, remap_neighbors, neighbors_to_adjacency_torch
from dinov2.data.datasets.data_utils import connect_graph, remove_axon, rotate_cell
import json
import copy
import json
import torch
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE

from dinov2.data.datasets.neuron_morpho import NeuronMorpho
from dinov2.train.utils_graph import plot_neuron, plot_tsne, neighbors_to_adjacency_torch, compute_eig_lapl_torch_batch
from dinov2.models import build_model_from_cfg
import numpy as np
from dinov2.utils.config import setup
from dinov2.train.train import get_args_parser,build_optimizer,build_schedulers
from dinov2.train.ssl_meta_arch import SSLMetaArch
import os
from dinov2.fsdp import FSDPCheckpointer
from dinov2.models.graphdino import GraphTransformer
import torch.optim as optim
from torch.utils.data import random_split
import datetime

class Condition_dataset(Dataset):

    def __init__(
        self,
        data_path,
        keep_node,
        filling,
    ) -> None:
        
        cell_ids = list(np.load(Path(data_path, 'all_ids.npy')))
        cell_labels = list(np.load(Path(data_path, 'all_labels.npy')))
        
        self.manager = Manager()
        self.cells = self.manager.dict()

        count = 0
        for cell_id in tqdm(cell_ids):
            
            soma_id = 0

            features = np.load(Path(data_path, 'skeletons', str(cell_id), 'features.npy'))
            with open(Path(data_path, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                neighbors = pickle.load(f)
            
            neighbors, not_deleted = subsample_graph(neighbors=neighbors, not_deleted=set(range(len(neighbors))), keep_nodes=keep_node,  protected=[soma_id], filling=filling)

            neighbors, subsampled2new = remap_neighbors(neighbors)
            soma_id = subsampled2new[soma_id]

            if filling:
                if len(features) < keep_node:
                    features = np.concatenate( (features, [[ -1.0 for i in range(features.shape[-1])] for j in range(keep_node - len(features))]), axis=0)
            
            features = features[list(subsampled2new.keys()), :8]
            item = {'features': features,  'neighbors': neighbors, 'labels': cell_labels[cell_id] }
            
            self.cells[count] = item
            count += 1
    
    def __len__(self):
        return len(self.cells)

    def __getitem__(self, index): 
        cell = self.cells[index]
        neigh = cell['neighbors']

        adj = neighbors_to_adjacency_torch(neigh, list(neigh.keys())).float().cuda()

        feat = torch.tensor(cell['features'], dtype=torch.float) 
        label = torch.tensor(cell['labels'], dtype=torch.long) 

        return feat, adj, label

class Classifier(nn.Module):
    def __init__(self, backbone, num_classes, dim, proj_dim):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, num_classes)
        )
        
    def forward(self, feat, adj, lapl):
        x = self.backbone(feat, adj, lapl)["x_norm_clstoken"]
        x = self.classifier(x.float())
        return x


root_dir = '/mnt/data/aim/liyaxuan/projects/project2/test_havemask/eval/0.132433/'
iter = '11594'
batch_sum = 32
filled = True

config_path = root_dir + '../../../configs/config_condition_classification.json'

args = get_args_parser(add_help=True).parse_args()
args.config_file = root_dir + '../../../dinov2/configs/train/test.yaml'
cfg = setup(args)
config = json.load(open(config_path))

out_feature_dim = config['model']['dim']
proj_feature_dim = config['model']['proj_dim']
num_classes = config['model']['num_classes']

backbonemodel = SSLMetaArch(cfg).to(torch.device("cuda"))
backbonemodel.prepare_for_distributed_training()

student_state_dict = torch.load(root_dir + iter + '_student_checkpoint.pth')
teacher_state_dict = torch.load(root_dir + iter + '_teacher_checkpoint.pth')

backbonemodel.student.load_state_dict(student_state_dict["student"])
backbonemodel.teacher.load_state_dict(teacher_state_dict["teacher"])

dset = Condition_dataset(data_path=root_dir + '../../../condition_processed/10.1002_admi.201700819/',keep_node=config['data']['keep_node'], filling = filled)

model_classifier = Classifier(backbone=backbonemodel.student.backbone, num_classes=num_classes, dim=out_feature_dim, proj_dim=proj_feature_dim).cuda() # Assuming backbone.student outputs feature vectors
optimizer = optim.Adam(model_classifier.parameters())
criterion = nn.CrossEntropyLoss()

train_size = int(0.8 * len(dset))
val_size = len(dset) - train_size

train_dataset, val_dataset = random_split(dset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_sum, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_sum, shuffle=False)

num_epochs = 50

nowdate = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
logger = logging.getLogger('class_logger')
logger.setLevel(logging.DEBUG)
file_log = logging.FileHandler(config['logging']['path'] + nowdate,'a',encoding='utf-8')
file_log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s ')
file_log.setFormatter(formatter)
logger.addHandler(file_log)

for epoch in range(num_epochs):

    running_loss = 0.0
    ACC = 0
    sum_num = 0

    for feat, adj, label in train_loader:

        lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=config['model']['pos_dim']).float().cuda()

        feat = feat.cuda()
        adj = adj.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        outputs = model_classifier(feat, adj, lapl)

        max_positions = torch.argmax(outputs, dim=1)
        corrects = (max_positions == label)
        answer = corrects.sum().item()
        ACC += answer

        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        sum_num += len(feat)

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)} Acc: { ACC * 100.0 / sum_num}%")
    logger.info('Epoch {} | Train: Loss {:.4f} | Acc {:.2f}%'.format(epoch+1, running_loss/len(train_loader), ACC * 100.0 / sum_num))
    
    running_loss = 0.0
    ACC = 0
    sum_num = 0

    for feat, adj, label in val_loader:

        lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=config['model']['pos_dim']).float().cuda()

        feat = feat.cuda()
        adj = adj.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        outputs = model_classifier(feat, adj, lapl)

        max_positions = torch.argmax(outputs, dim=1)
        corrects = (max_positions == label)
        answer = corrects.sum().item()
        ACC += answer

        loss = criterion(outputs, label)

        running_loss += loss.item()
        sum_num += len(feat)

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(val_loader)} Acc: { ACC * 100.0 / sum_num}%")
    logger.info('Epoch {} | Val: Loss {:.4f} | Acc {:.2f}%'.format(epoch+1, running_loss/len(val_loader), ACC * 100.0 / sum_num))

print('Finished Training')
