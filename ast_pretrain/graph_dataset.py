import time
import math
import operator

import dgl
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from dgl.data import AmazonCoBuy, Coauthor
# from dgl.nodeflow import NodeFlow
import json
from operator import itemgetter
import os
from Util import *
from Trans_funcUtil import invokeTransformations
import scipy.sparse as sparse
import torch.nn.functional as F
from scipy.sparse import linalg
import sklearn.preprocessing as preprocessing

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # 获取当前job的查询和图和代码
    dataset.cur_graphs = [dataset.graphs[i] for i in dataset.jobs[worker_id]]
    dataset.cur_querys = [dataset.query_seq_list[i] for i in dataset.jobs[worker_id]]
    dataset.cur_codes = [dataset.codes[i] for i in dataset.jobs[worker_id]]

    # 之前是节点数，现在就直接计算图的数量就可以
    dataset.length = len(dataset.jobs[worker_id])

def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64") # ?????随机?同一个图的特征值会变不一样的啊你妈的
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0) # 特征值 特征向量
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float) # 邻接矩阵
    # 每个点的入度，做成一个对角矩阵
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    # 拉普拉斯矩阵就是这么乘出来的
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    g.ndata["seed"] = torch.zeros(g.number_of_nodes(), dtype=torch.long)
    g.ndata["seed"][0] = 1
    return g


class LoadBalanceGraphDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            num_workers=1,
            data_path="data/pretrain/",
            num_samples=10000,
            num_copies=1,
    ):
        super(LoadBalanceGraphDataset).__init__()
        self.data_path = data_path
        self.num_samples = num_samples
        self.query_vocab, self.ast_vocab = get_token_vocab(data_path)
        self.query_seq_list, self.codes, self.var_vocab, self.graphs = get_train_data(self.query_vocab, self.ast_vocab, data_path)
        self.num_workers = num_workers
        self.graph_sizes = []
        for graph in self.graphs:
            self.graph_sizes.append(graph.batch_num_nodes())

        print("load dataset done")

        assert num_workers % num_copies == 0
        jobs = [list() for i in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(self.graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples * num_workers 

    def __len__(self):
        return self.num_samples * self.num_workers

    def __iter__(self):
        prob = np.ones(self.length) / self.length
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob
        )
        for idx in samples:
            yield self.__getitem__(idx)

    # 生成k和q
    def __getitem__(self, idx):

        query = self.cur_querys[idx]

        graph_q = self.cur_graphs[idx]
        graph_q = _add_undirected_graph_positional_embedding(graph_q, self.positional_embedding_size)
        
        
        code = self.cur_codes[idx]

        try:
            code_new = invokeTransformations(code, self.var_vocab)
            tree = ast.parse(code_new).body[0]
            # 刚生成的tree里的节点是没有id的, 遍历一下加上id
            get_ast_seq_level(tree)
            graph_k = get_ast_graph(tree, self.ast_vocab)
            graph_k = _add_undirected_graph_positional_embedding(graph_k, self.positional_embedding_size)
            return query, graph_q, graph_k

        except:
            return query, graph_q, graph_q



