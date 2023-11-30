from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
import networkx as nx
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.core import swc
import numpy as np
from dinov2.data.datasets.data_utils import connect_graph, rotate_cell
from dinov2.data.datasets.utils import neighbors_to_adjacency, plot_neuron, compute_node_distances, adjacency_to_neighbors
import os
import numpy as np

import pandas as pd
from tqdm import *
import numpy as np
import networkx as nx

import logging
import os
import time
import json

class Config():
    # 创建logger实例
    logger = logging.getLogger()
    # 设置logger的日志级别
    logger.setLevel(logging.DEBUG)

    # 添加控制台管理器(即控制台展示log内容）
    ls = logging.StreamHandler()
    ls.setLevel(logging.DEBUG)

    # 设置log的记录格式
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(filename)s-[line:%(lineno)d]''-%(levelname)s: %(message)s')

    # 把格式添加到控制台管理器,即控制台打印日志
    ls.setFormatter(formatter)
    # 把控制台添加到logger
    logger.addHandler(ls)

    # 先在项目目录下建一个logs目录，来存放log文件（可自定义路径）
    logdir = os.path.join('/mnt/data/aim/liyaxuan/projects/project2', 'logs')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    # 再在logs目录下创建以日期开头的.log文件
    logfile = os.path.join(logdir, time.strftime('%Y-%m-%d %H:%M:%S') + '.log')

    # 添加log的文件处理器，并设置log的配置文件模式编码
    lf = logging.FileHandler(filename=logfile, encoding='utf8')
    # 设置log文件处理器记录的日志级别
    lf.setLevel(logging.DEBUG)
    # 设置日志记录的格式
    lf.setFormatter(formatter)
    # 把文件处理器添加到log
    logger.addHandler(lf)

    def get_config(self):
        return self.logger

#这里创建了一个logger对象，其他文件使用时，只需导入即可，如from common.log.logger import logger
logger = Config().get_config()

    
def remove_axon(neighbors, features, adj_matrix, soma_id):
    
    axon_mask = (features[:, 5] == 1)

    axon_idcs = list(np.where(axon_mask)[0])

    non_key_nodes = [k for k, v in neighbors.items() if len(v) in [1, 2] and k in axon_idcs]
    
    if soma_id in non_key_nodes:
        non_key_nodes.remove(soma_id)

    G = nx.Graph(adj_matrix)
    
    for node in non_key_nodes:
        neighs = list(G.neighbors(node))

        if node in neighs:
            neighs.remove(node)

        G.remove_node(node)
        axon_idcs.remove(node)
        
        if len(neighs) == 2:
            if not nx.has_path(G, neighs[0], neighs[1]):
                G.add_edge(neighs[0], neighs[1])

    adj_matrix = nx.to_numpy_array(G)
    
    neighbors = adjacency_to_neighbors(adj_matrix)
    
    mapping = {i: j for j, i in enumerate(sorted(set(range(features.shape[0])) - set(non_key_nodes)))}
    
    features = np.delete(features, non_key_nodes, axis=0)
    
    soma_id = mapping[soma_id]
    
    not_deleted = list(mapping.values())
    adj_matrix = neighbors_to_adjacency(neighbors, not_deleted)

    return neighbors, features, soma_id


all_ids = []
all_labels = []

root_dir = '/mnt/data/aim/liyaxuan/projects/project2/condition_processed/10.1002_admi.201700819/'

with os.scandir(root_dir) as entries:
    all_swc_name = [entry.name for entry in entries if (entry.is_file() and entry.name[-4:] == '.swc') ]


filename_to_phenotype = json.load(open(root_dir + 'label/filename_to_phenotype.json'))
phenotype_to_label = json.load(open(root_dir + 'label/phenotype_to_label.json'))

cell_allids = []
cell_alllabels = []
now_num = 0

for swc_name in tqdm(all_swc_name):
    
    swc_path = root_dir  + str(swc_name)
    
    try:
        from io import StringIO

        with open(swc_path, 'r') as f:
            lines = [line for line in f if not line.lstrip().startswith('#') and line.strip() != '']

        virtual_file = StringIO('\n'.join(lines))

        morphology = pd.read_csv(
            virtual_file,
            delim_whitespace=True,
            skipinitialspace=True,
            names=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'],
            index_col=False
        )

        have_err_read = 0
        
        for i in range(morphology.shape[0]):
            item = morphology.iloc[i]
            for name in ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']:
                if isinstance(item[name], str):
                    have_err_read = 1
                    all_errors.append(now_id)
                    break
            if have_err_read == 1:
                break
        
        if have_err_read == 1:
            continue

    except:
        continue
    
    soma = morphology.iloc[0]
    
    soma_pos = np.array([soma['x'], soma['y'], soma['z']])
    soma_id = int(soma['id'] - 1)

    # # Process graph.
    neighbors = {}
    idx2node = {}
    # hav_err = 0
    
    for i in range(morphology.shape[0]):
        # Get node features.
        
        item = morphology.iloc[i]
        sec_type = [0, 0, 0, 0, 0, 0, 0]

        if item['type'] > 7:
            item['type'] = 7
        sec_type[int(item['type']) - 1] = 1
        feat = tuple([item['x'], item['y'], item['z'], item['radius']]) + tuple(sec_type)
        idx2node[i] = feat
        
        # Get neighbors.
        neighbors[i] = set(morphology[morphology['parent']==item['id']]['id'])
        neighbors[i] = set([int(i-1) for i in neighbors[i]])
        if item['parent'] != -1:
            neighbors[i].add(item['parent'] - 1)

    features = np.array(list(idx2node.values()))
    
    if np.any(np.isnan(features)):
        all_errors.append(now_id)
        continue
        
    # Normalize soma position to origin.
    norm_features = features.copy()
    norm_features[:, :3] = norm_features[:, :3] - soma_pos

    adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))

    G = nx.Graph(adj_matrix)
    
    if nx.number_connected_components(G) > 1:
        all_errors.append(now_id)
        continue
    
    assert len(neighbors) == len(adj_matrix)
    
    # Remove axons.
    neighbors, norm_features, soma_id = remove_axon(neighbors, norm_features, adj_matrix, int(soma_id))
    
    adj_matrix = neighbors_to_adjacency(neighbors, range(len(neighbors)))

    distances = compute_node_distances(soma_id, neighbors)

    keys = [ key for key in neighbors]
    
    assert len(neighbors) == len(norm_features)
    assert ~np.any(np.isnan(norm_features))
    
    path = Path(root_dir + 'skeletons/', str(now_num))
    path.mkdir(parents=True, exist_ok=True)
    
    np.save(Path(path, 'features'), norm_features)

    with open(Path(path, 'neighbors.pkl'), 'wb') as f:
        pickle.dump(dict(neighbors), f, pickle.HIGHEST_PROTOCOL)

    cell_allids.append(now_num)
    cell_alllabels.append(int(phenotype_to_label[filename_to_phenotype[swc_name]]))
    now_num += 1

np.save(root_dir + 'all_ids.npy', cell_allids)
np.save(root_dir + 'all_labels.npy', cell_alllabels)

