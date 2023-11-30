import os
import json
import torch
import logging
import argparse
import numpy as np 
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from pathlib import Path
from torch.multiprocessing import Manager

from ssl_neuron.ssl_neuron.utils import subsample_graph, rotate_graph, jitter_node_pos, translate_soma_pos, get_leaf_branch_nodes, compute_node_distances, drop_random_branch, remap_neighbors, neighbors_to_adjacency_torch
from ssl_neuron.ssl_neuron.utils import AverageMeter, compute_eig_lapl_torch_batch

# Attention and Block adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# DINO adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/dino.py

import copy
import torch
import torch.nn as nn
from typing import Any
import math

class Adapter(nn.Module):
    def __init__(self,
                 dropout=0.0,
                 n_embd = None):
        
        super().__init__()

        self.n_embd = 768 if n_embd is None else n_embd

        self.down_size = 64

        self.scale = nn.Parameter(torch.ones(1))

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        return up
    
class GraphAttention(nn.Module):
    """ Implements GraphAttention.

    Graph Attention interpolates global transformer attention
    (all nodes attend to all other nodes based on their
    dot product similarity) and message passing (nodes attend
    to their 1-order neighbour based on dot-product
    attention).

    Attributes:
        dim: Dimensionality of key, query and value vectors.
        num_heads: Number of parallel attention heads.
        bias: If set to `True`, use bias in input projection layers.
          Default is `False`.
        use_exp: If set to `True`, use the exponential of the predicted
          weights to trade-off global and local attention.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 bias: bool = False,
                 use_exp: bool = True) -> nn.Module:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.use_exp = use_exp

        self.qkv_projection = nn.Linear(dim, dim * num_heads * 3, bias=bias)
        self.proj = nn.Linear(dim * num_heads, dim)
        
        # Weigth to trade of local vs. global attention.
        self.predict_gamma = nn.Linear(dim, 2)
        # Initialize projection such that gamma is close to 1
        # in the beginning of training.
        self.predict_gamma.weight.data.uniform_(0.0, 0.01)

        
    @torch.jit.script
    def fused_mul_add(a, b, c, d):
        return (a * b) + (c * d)

    def forward(self, x, adj):
        B, N, C = x.shape # (batch x num_nodes x feat_dim)
        qkv = self.qkv_projection(x).view(B, N, 3, self.num_heads, self.dim).permute(0, 3, 1, 2, 4)
        query, key, value = qkv.unbind(dim=3) # (batch x num_heads x num_nodes x dim)

        attn = (query @ key.transpose(-2, -1)) * self.scale # (batch x num_heads x num_nodes x num_nodes)

        # Predict trade-off weight per node
        gamma = self.predict_gamma(x)[:, None].repeat(1, self.num_heads, 1, 1)
        if self.use_exp:
            # Parameterize gamma to always be positive
            gamma = torch.exp(gamma)

        adj = adj[:, None].repeat(1, self.num_heads, 1, 1)

        # Compute trade-off between local and global attention.
        attn = self.fused_mul_add(gamma[:, :, :, 0:1], attn, gamma[:, :, :, 1:2], adj)
        
        attn = attn.softmax(dim=-1)

        x = (attn @ value).transpose(1, 2).reshape(B, N, -1) # (batch_size x num_nodes x (num_heads * dim))
        return self.proj(x)
    
    
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> nn.Module:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """ Implements an attention block.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 bias: bool = False,
                 use_exp: bool = True,
                 norm_layer: Any = nn.LayerNorm) -> nn.Module:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GraphAttention(dim, num_heads=num_heads, bias=bias, use_exp=use_exp)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim=dim, hidden_dim=dim * mlp_ratio)
        self.adapter = Adapter(n_embd=dim)

    def forward(self, x, a):

        lst_x = x
        x = self.norm1(x)
        x = self.attn(x, a) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        x = x + self.adapter(lst_x)
        return x
    
    
class GraphTransformer(nn.Module):
    def __init__(self,
                 n_nodes: int = 200,
                 dim: int = 32,
                 depth: int = 5,
                 num_heads: int = 8,
                 mlp_ratio: int = 2,
                 feat_dim: int = 8,
                 num_classes: int = 1000,
                 pos_dim: int = 32,
                 proj_dim: int = 128,
                 use_exp: bool = True) -> nn.Module:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.Sequential(*[
            AttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_exp=use_exp)
            for i in range(depth)])

        self.to_pos_embedding = nn.Linear(pos_dim, dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        self.projector = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, num_classes)
        )

        self.to_node_embedding = nn.Sequential(
            nn.Linear(feat_dim, dim * 2),
            nn.ReLU(True),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, node_feat, adj, lapl):
        B, N, _ = node_feat.shape

        # Compute initial node embedding.
        x = self.to_node_embedding(node_feat)

        # Compute positional encoding
        pos_embedding_token = self.to_pos_embedding(lapl)

        # Add "classification" token
        cls_pos_enc = self.cls_pos_embedding.repeat(B, 1, 1)
        pos_embedding = torch.cat((cls_pos_enc, pos_embedding_token), dim=1)

        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add classification token entry to adjanceny matrix. 
        adj_cls = torch.zeros(B, N + 1, N + 1, device=node_feat.device)
        # TODO(test if useful)
        adj_cls[:, 0, 0] = 1.
        adj_cls[:, 1:, 1:] = adj

        x += pos_embedding
        for block in self.blocks:
            x = block(x, adj_cls)
        x = x[:, 0]
        x = self.mlp_head(x)

        return x, self.projector(x)


class ExponentialMovingAverage():
    """ Exponential moving average.

    Attributes:
        decay: Moving average decay parameter in [0., 1.] (float).
    """
    def __init__(self, decay: float):
        super().__init__()
        self.decay = decay
        assert (decay > 0.) and (decay < 1.), 'Decay must be in [0., 1.]'

    def update_average(
        self,
        previous_state: torch.Tensor,
        update: torch.Tensor,
        decay: float = None,
    ):
        if previous_state is None:
            return update
        if decay is not None:
            return previous_state * decay + (1 - decay) * update
        else:
            return previous_state * self.decay + (1 - self.decay) * update


def update_moving_average(ema_updater, teacher_model, student_model, decay=None):
    for student_params, teacher_params in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_weights, weight_update = teacher_params.data, student_params.data
        teacher_params.data = ema_updater.update_average(teacher_weights, weight_update, decay=decay)


class GraphDINO(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        num_classes: int = 1000,
        student_temp: float = 0.9,
        teacher_temp: float = 0.06,
        moving_average_decay: float = 0.999,
        center_moving_average_decay: float = 0.9,
    ):
        super().__init__()

        self.student_encoder = transformer
        self.teacher_encoder = copy.deepcopy(self.student_encoder)

        # Weights of teacher model are updated using an exponential moving
        # average of the student model. Thus, disable gradient update.
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False

        self.teacher_ema_updater = ExponentialMovingAverage(moving_average_decay)

        self.register_buffer('teacher_centers', torch.zeros(1, num_classes))
        self.register_buffer('previous_centers',  torch.zeros(1, num_classes))

        self.teacher_centering_ema_updater = ExponentialMovingAverage(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def compute_loss(self, teacher_logits, student_logits, eps = 1e-20):
        teacher_logits = teacher_logits.detach()
        student_probs = (student_logits / self.student_temp).softmax(dim = -1)
        teacher_probs = ((teacher_logits - self.teacher_centers) / self.teacher_temp).softmax(dim = -1)
        loss = - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()
        return loss

    def update_moving_average(self, decay=None):
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder, decay=decay)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.previous_centers)
        self.teacher_centers.copy_(new_teacher_centers)

    def forward(self, node_feat1, node_feat2, adj1, adj2, lapl1, lapl2):
        batch_size = node_feat1.shape[0]

        # Concatenate the two views to compute embeddings as one batch.
        node_feat = torch.cat([node_feat1, node_feat2], dim=0)
        adj = torch.cat([adj1, adj2], dim=0)
        lapl = torch.cat([lapl1, lapl2], dim=0)

        _, student_proj = self.student_encoder(node_feat, adj, lapl)
        student_proj1, student_proj2 = torch.split(student_proj, batch_size, dim=0)

        with torch.no_grad():
            _, teacher_proj = self.teacher_encoder(node_feat, adj, lapl)
            teacher_proj1, teacher_proj2 = torch.split(teacher_proj, batch_size, dim=0)

        teacher_logits_avg = teacher_proj.mean(dim = 0)
        self.previous_centers.copy_(teacher_logits_avg)

        loss1 = self.compute_loss(teacher_proj1, student_proj2)
        loss2 = self.compute_loss(teacher_proj2, student_proj1)
        loss = (loss1 + loss2) / 2

        return loss


def create_model(config):
    num_classes = config['model']['num_classes']

    # Create encoder.
    transformer = GraphTransformer(n_nodes=config['data']['n_nodes'],
                 dim=config['model']['dim'], 
                 depth=config['model']['depth'], 
                 num_heads=config['model']['n_head'],
                 feat_dim=config['data']['feat_dim'],
                 pos_dim=config['model']['pos_dim'],
                 num_classes=num_classes)

    # Create GraphDINO.
    model = GraphDINO(transformer,
                 num_classes=num_classes, 
                 moving_average_decay=config['model']['move_avg'],
                 center_moving_average_decay=config['model']['center_avg'],
                 teacher_temp=config['model']['teacher_temp']
                )
    
    for name, p in model.named_parameters():
          if 'adapter' in name:
            #print('name: ',name)
            p.requires_grad = True
          else:
            p.requires_grad = False      

    return model

class CellGraphDataset(Dataset):
    def __init__(self, config, mode='train', inference=False, num_class = None):

        self.config = config
        self.mode = mode
        self.inference = inference
        data_path = config['data']['path']

        # Augmentation parameters.
        self.jitter_var = config['data']['jitter_var']
        self.rotation_axis = config['data']['rotation_axis']
        self.n_drop_branch = config['data']['n_drop_branch']
        self.translate_var = config['data']['translate_var']
        self.n_nodes = config['data']['n_nodes']

        tmp_list = np.load(Path(data_path, f'{mode}_data.npy'))

        # Load graphs.
        self.manager = Manager()
        self.cells = self.manager.dict()
        
        label_tmp = []

        count = 0
        for tp in tqdm(tmp_list):
            
            cell_id = tp[0]
            cell_label = tp[1]

            soma_id = 0

            features = np.load(Path(data_path, 'skeletons', str(cell_id), 'features.npy'))
            with open(Path(data_path, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                neighbors = pickle.load(f)
            
            assert len(features) == len(neighbors)

            if len(features) >= self.n_nodes or self.inference:
                
                
                # Subsample graphs for faster processing during training.
                neighbors, not_deleted = subsample_graph(neighbors=neighbors, 
                                                         not_deleted=set(range(len(neighbors))), 
                                                         keep_nodes=1000, 
                                                         protected=[soma_id])
                # Remap neighbor indices to 0..999.
                neighbors, subsampled2new = remap_neighbors(neighbors)
                soma_id = subsampled2new[soma_id]

                # Accumulate features of subsampled nodes.
                features = features[list(subsampled2new.keys()), :3]

                leaf_branch_nodes = get_leaf_branch_nodes(neighbors)
                
                # Using the distances we can infer the direction of an edge.
                distances = compute_node_distances(soma_id, neighbors)

                item = {
                    'cell_id': cell_id,
                    'features': features, 
                    'neighbors': neighbors,
                    'distances': distances,
                    'soma_id': soma_id,
                    'leaf_branch_nodes': leaf_branch_nodes,
                }

                not_connect = 0
                for key in neighbors:
                    for to in neighbors[key]:
                        if to not in distances.keys():
                            not_connect = 1
                            break
                    if not_connect == 1:
                        break
                
                if not_connect == 1:
                    continue

                self.cells[count] = item
                label_tmp.append(cell_label)
                count += 1

        self.labels = nn.functional.one_hot(torch.tensor(label_tmp), num_classes=num_class)
        self.num_samples = len(self.cells)

    def __len__(self):
        return self.num_samples

    def _delete_subbranch(self, neighbors, soma_id, distances, leaf_branch_nodes):

        leaf_branch_nodes = set(leaf_branch_nodes)
        not_deleted = set(range(len(neighbors))) 
        for i in range(self.n_drop_branch):
            neighbors, drop_nodes = drop_random_branch(leaf_branch_nodes, neighbors, distances, keep_nodes=self.n_nodes)
            not_deleted -= drop_nodes
            leaf_branch_nodes -= drop_nodes
            
            if len(leaf_branch_nodes) == 0:
                break

        return not_deleted

    def _reduce_nodes(self, neighbors, soma_id, distances, leaf_branch_nodes):
        neighbors2 = {k: set(v) for k, v in neighbors.items()}

        # Delete random branches.
        not_deleted = self._delete_subbranch(neighbors2, soma_id, distances, leaf_branch_nodes)

        # Subsample graphs to fixed number of nodes.
        neighbors2, not_deleted = subsample_graph(neighbors=neighbors2, not_deleted=not_deleted, keep_nodes=self.n_nodes, protected=soma_id)

        # Compute new adjacency matrix.
        adj_matrix = neighbors_to_adjacency_torch(neighbors2, not_deleted)
        
        assert adj_matrix.shape == (self.n_nodes, self.n_nodes), '{} {}'.format(adj_matrix.shape)
        
        return neighbors2, adj_matrix, not_deleted
    
    
    def _augment_node_position(self, features):
        # Extract positional features (xyz-position).
        pos = features[:, :3]

        # Rotate (random 3D rotation or rotation around specific axis).
        rot_pos = rotate_graph(pos, axis=self.rotation_axis)

        # Randomly jitter node position.
        jittered_pos = jitter_node_pos(rot_pos, scale=self.jitter_var)
        
        # Translate neuron position as a whole.
        jittered_pos = translate_soma_pos(jittered_pos, scale=self.translate_var)
        
        features[:, :3] = jittered_pos

        return features
    
    def _augment(self, cell):

        features = cell['features']
        neighbors = cell['neighbors']
        distances = cell['distances']

        # Reduce nodes to N == n_nodes via subgraph deletion and subsampling.
        neighbors2, adj_matrix, not_deleted = self._reduce_nodes(neighbors, [int(cell['soma_id'])], distances, cell['leaf_branch_nodes'])

        # Extract features of remaining nodes.
        new_features = features[not_deleted].copy()
       
        # Augment node position via rotation and jittering.
        new_features = self._augment_node_position(new_features)

        return new_features, adj_matrix
    
    def __getitem__(self, index): 

        cell = self.cells[index]
        label = self.labels[index]

        features, adj_matrix = self._augment(cell)

        return features, adj_matrix, label
    

def build_dataloader(config, num_class, use_cuda=torch.cuda.is_available()):

    kwargs = {'num_workers':config['data']['num_workers'], 'pin_memory':True, 'persistent_workers': True} if use_cuda else {}

    train_loader = DataLoader(
            CellGraphDataset(config, mode='train', num_class = num_class),
            batch_size=config['data']['batch_size'], 
            shuffle=True, 
            drop_last=True,
            **kwargs)

    val_dataset = CellGraphDataset(config, mode='val', num_class = num_class)
    batch_size = val_dataset.num_samples if val_dataset.__len__() < config['data']['batch_size'] else config['data']['batch_size']
    val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            **kwargs)

    return train_loader, val_loader

class Trainer(object):
    def __init__(self, config, model, dataloaders):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config

        #for param in self.model.backbone.parameters():
        #    param.requires_grad = False

        import datetime, os
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.ckpt_dir = config['trainer']['ckpt_dir'] + now
        os.mkdir(self.ckpt_dir)

        self.logger = logging.getLogger('class_logger')
        self.logger.setLevel(logging.DEBUG)
        file_log = logging.FileHandler(config['logging']['path'] + now,'a',encoding='utf-8')
        file_log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(message)s ')
        file_log.setFormatter(formatter)
        self.logger.addHandler(file_log)

        self.save_every = config['trainer']['save_ckpt_every']

        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader= dataloaders[1]

        ### trainings params
        self.max_iter = config['optimizer']['max_iter']
        self.backbone_initlr = config['optimizer']['backbone_lr']
        self.classifier_initlr = config['optimizer']['classifier_lr']
        self.exp_decay = config['optimizer']['exp_decay']

        self.backbone_warmup = torch.linspace(0., self.backbone_initlr,  steps=(self.max_iter // 50)+1)[1:]
        self.classifier_warmup = torch.linspace(0., self.classifier_initlr,  steps=(self.max_iter // 50)+1)[1:]
        self.lr_decay = self.max_iter // 5
        
        self.backbone_optimizer = optim.Adam(list(self.model.backbone.parameters()), lr=0)
        self.classifier_optimizer = optim.Adam(list(self.model.classifier.parameters()), lr=0)

        self.loss = nn.CrossEntropyLoss()
        
        
    def set_lr(self): 

        if self.curr_iter < len(self.backbone_warmup):
            backbone_lr = self.backbone_warmup[self.curr_iter]
        else:
            backbone_lr = self.backbone_initlr * self.exp_decay ** ((self.curr_iter - len(self.backbone_warmup)) / self.lr_decay)
        
        if self.curr_iter < len(self.classifier_warmup):
            classifier_lr = self.classifier_warmup[self.curr_iter]
        else:
            classifier_lr = self.classifier_initlr * self.exp_decay ** ((self.curr_iter - len(self.classifier_warmup)) / self.lr_decay)
        
        for param_group in self.backbone_optimizer.param_groups:
            param_group['lr'] = backbone_lr

        for param_group in self.classifier_optimizer.param_groups:
            param_group['lr'] = classifier_lr
        
        return backbone_lr, classifier_lr
        

    def train(self):     
        self.curr_iter = 0
        epoch = 0
        while self.curr_iter < self.max_iter:
            # Run one epoch.
            self._train_epoch(epoch)

            if epoch % self.save_every == 0:
                # Save checkpoint.
                self._save_checkpoint(epoch)
            
            epoch += 1


    def _train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        acc = 0
        total = 0

        for data in self.train_loader:

            feat, adj, label = [x.float().to(self.device, non_blocking=True) for x in data]

            lap = compute_eig_lapl_torch_batch(adj)
            n = adj.shape[0]
            
            self.set_lr()
            self.backbone_optimizer.zero_grad(set_to_none=True)
            self.classifier_optimizer.zero_grad(set_to_none=True)

            ans_class = self.model(feat, adj, lap)

            label_pre = ans_class.argmax(dim=1)
            label_ans = label.argmax(dim = 1)
            acc += sum((label_ans == label_pre) + 0)
            total += len(label)

            loss = self.loss(ans_class, label)

            # optimize 
            loss.sum().backward()

            self.backbone_optimizer.step()
            self.classifier_optimizer.step()

            losses.update(loss.detach(), n)
            self.curr_iter += 1

        print('Epoch {} | Train: Loss {:.4f} | Acc {:.4f}'.format(epoch, losses.avg, acc * 1.0 / total))
        self.logger.info('Epoch {} | Train: Loss {:.4f} | Acc {:.4f}'.format(epoch, losses.avg, acc * 1.0 / total))
        self.val(epoch)

    def val(self, epoch_num):     
        
        acc = 0
        total = 0
        losses = AverageMeter()

        for data in self.val_loader:

            feat, adj, label = [x.float().to(self.device, non_blocking=True) for x in data]

            lap = compute_eig_lapl_torch_batch(adj)
            n = adj.shape[0]
            
            ans_class = self.model(feat, adj, lap)

            label_pre = ans_class.argmax(dim=1)
            label_ans = label.argmax(dim = 1)
            acc += sum((label_ans == label_pre) + 0)
            total += len(label)

            loss = self.loss(ans_class, label)

            losses.update(loss.detach(), n)
            self.curr_iter += 1

        print('Epoch {} | Val: Loss {:.4f} | Acc {:.4f}'.format(epoch_num, losses.avg, acc * 1.0 / total))
        self.logger.info('Epoch {} | Val: Loss {:.4f} | Acc {:.4f}'.format(epoch_num, losses.avg, acc * 1.0 / total))


    def _save_checkpoint(self, epoch):
        filename = 'ckpt_{}.pt'.format(epoch)
        PATH = os.path.join(self.ckpt_dir, filename)
        torch.save(self.model.state_dict(), PATH)
        print('Save model after epoch {} as {}.'.format(epoch, filename))


class Downstream_Classification(nn.Module):
    def __init__(self, model, num_class, ckpt_path):
        super().__init__()
        self.backbone = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
        self.classifier = nn.Linear(32, num_class)
        state_dict = torch.load(ckpt_path)
        self.backbone.load_state_dict(state_dict, strict=False)
        self.softmax = nn.Softmax()

        #for name, p in self.backbone.named_parameters():
        #    print('name:', name, ' state:', p.requires_grad)

    def forward(self, feat, adj, lapl):
        encode = self.backbone.module.student_encoder(feat, adj, lapl)
        ans = self.classifier(encode[0])
        ans = self.softmax(ans)
        return ans



clss = ['L1_DAC', 'L1_DLAC', 'L1_HAC', 'L1_NGC-DA', 'L1_NGC-SA', 'L1_SLAC', 'L23_BP', 'L23_BTC', 'L23_ChC', 'L23_DBC', 'L23_LBC', 'L23_MC', 'L23_NBC', 'L23_NGC', 'L23_PC', 'L23_SBC', 'L4_BP', 'L4_BTC', 'L4_ChC', 'L4_DBC', 'L4_LBC', 'L4_MC', 'L4_NBC', 'L4_NGC', 'L4_SBC', 'L4_PC', 'L4_SP', 'L4_SS', 'L5_BP', 'L5_BTC', 'L5_ChC', 'L5_DBC', 'L5_LBC', 'L5_MC', 'L5_NBC', 'L5_NGC', 'L5_SBC', 'L5_STPC', 'L5_TTPC1', 'L5_TTPC2', 'L5_UTPC', 'L6_BPZ', 'L6_BPC', 'L6_BTC', 'L6_ChC', 'L6_DBC', 'L6_IPC', 'L6_LBC', 'L6_MC', 'L6_NBC', 'L6_NGC', 'L6_SBC', 'L6_TPC_L1', 'L6_TPC_L4', 'L6_UTPC']

num_class = len(clss)

# load config
ckpt_path = '/mnt/data/aim/liyaxuan/projects/project2/codebase/ckpts/2023-08-07T19-31-07/ckpt_1030.pt'
config = json.load(open('/mnt/data/aim/liyaxuan/projects/project2/codebase/ssl_neuron/configs/config_classification_adapter.json'))

# load data
print('Loading dataset: {}'.format(config['data']['class']))
train_loader, val_loader = build_dataloader(config, num_class)

backbone = create_model(config)
model = Downstream_Classification(backbone, num_class, ckpt_path)


trainer = Trainer(config, model, [train_loader, val_loader])

print('Start training.')
trainer.train()
print('Done.')
