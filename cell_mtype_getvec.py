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

#now = 'std'
now = 'now'

def process(mode, cfg, model, config):

    dset = NeuronMorpho(root='', extra='', split=NeuronMorpho.Split['TRAIN'], config_path = config_path, mode=mode, inference=True)

    #print(dset.num_samples)
    latents = np.zeros((dset.num_samples, config['model']['dim']))
    
    print('dset.num_samples', dset.num_samples)
    for i in tqdm(range(dset.num_samples)):
        feat, neigh = dset.__getsingleitem__(i)
        adj = neighbors_to_adjacency_torch(neigh, list(neigh.keys())).float().cuda()[None, ]
        lapl = compute_eig_lapl_torch_batch(adj, pos_enc_dim=config['model']['pos_dim']).float().cuda()
        feat = torch.from_numpy(feat).float().cuda()[None, ]
        
        #latents[i] = model.forward(feat, adj, lapl)[0].cpu().detach()
        latents[i] = model.teacher.backbone(feat, adj, lapl)["x_norm_clstoken"].cpu().detach()

    np.save(root_dir + 'M1_EXC/' + f'{mode}_data.npy', latents)

root_dir = '/mnt/data/aim/liyaxuan/projects/project2/'
config_path = '/mnt/data/aim/liyaxuan/projects/project2/configs/config_mtype_classification.json'
config = json.load(open(config_path))

args = get_args_parser(add_help=True).parse_args()

cfg = setup(args)
model = SSLMetaArch(cfg).to(torch.device("cuda"))
model.prepare_for_distributed_training()

optimizer = build_optimizer(cfg, model.get_params_groups())
(
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    teacher_temp_schedule,
    last_layer_lr_schedule,
) = build_schedulers(cfg)

#print('resume:', resume)
# checkpointer
checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

#print('model_weights:', cfg.MODEL.WEIGHTS)
start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get("iteration", -1) + 1

model.cuda()

print('start_iter:', start_iter)

process('train', cfg, model, config)
process('val', cfg, model, config)

print('get_vec end')