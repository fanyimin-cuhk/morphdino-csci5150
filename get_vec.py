import json
import torch
from tqdm import tqdm
import seaborn as sns
from sklearn.manifold import TSNE

from ssl_neuron.ssl_neuron.datasets import GraphDataset
from ssl_neuron.ssl_neuron.utils import plot_neuron, plot_tsne, neighbors_to_adjacency_torch, compute_eig_lapl_torch_batch
from ssl_neuron.ssl_neuron.graphdino import create_model

import numpy as np

now = 'std1'

def process(mode):


    root_dir = '/mnt/data/aim/liyaxuan/projects/project2/'
    #config = json.load(open(root_dir + 'ssl_neuron/ssl_neuron/configs/config_mtype_classification.json'))
    config = json.load(open(root_dir + 'codebase/ssl_neuron/configs/config_mtype_classification.json'))
    model = create_model(config)
    
    if now == 'std':
        state_dict = torch.load(root_dir + '/ssl_neuron/ssl_neuron/ckpts/2023_08_04/ckpt.pt')
    else:
        torch.backends.cudnn.benchmark = True
        model = torch.nn.DataParallel(model)
        state_dict = torch.load(root_dir + 'codebase/ckpts/2023-08-07T19-31-07/ckpt_1030.pt')
   
    
    model.load_state_dict(state_dict)

    model.eval()
    model.cuda()

    dset = GraphDataset(config, mode=mode, inference = True)
    #print(dset.num_samples)
    latents = np.zeros((dset.num_samples, config['model']['dim']))

    for i in tqdm(range(dset.num_samples)):
        feat, neigh = dset.__getsingleitem__(i)
        adj = neighbors_to_adjacency_torch(neigh, list(neigh.keys())).float().cuda()[None, ]
        lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=config['model']['pos_dim']).float().cuda()
        feat = torch.from_numpy(feat).float().cuda()[None, ]
        
        if now == 'std':
            latents[i] = model.student_encoder.forward(feat, adj, lapl)[0].cpu().detach()
        else:
            latents[i] = model.module.student_encoder.forward(feat, adj, lapl)[0].cpu().detach()

    np.save(root_dir + 'M1_EXC/' + f'{mode}_data.npy', latents)

process('train')
process('val')