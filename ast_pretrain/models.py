#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_encoder.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/31 18:42
# TODO:

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Set2Set
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from gcc.models.gat import UnsupervisedGAT
from gin import UnsupervisedGIN



class QueryEncoder(nn.Module):
    def __init__(self, query_vocab_size, query_emb_size, query_lstm_size, query_hidden_size):
        super(QueryEncoder, self).__init__()
        self.pad_id = 0
        self.dropout = 0.25

        self.query_emb = nn.Embedding(num_embeddings=query_vocab_size, embedding_dim=query_emb_size, padding_idx=self.pad_id)
        self.query_dropout_layer = nn.Dropout(self.dropout)
        self.query_LSTM = nn.LSTM(query_emb_size, query_lstm_size, batch_first=True, bidirectional=True)
        self.query_linear = nn.Linear(2 * query_lstm_size, query_hidden_size)
        self.init_weights()

    def init_weights(self):
        for emb in [self.query_emb]:
            nn.init.uniform_(emb.weight, -0.1, 0.1)
            nn.init.constant_(emb.weight[0], 0)

        for p in [self.query_LSTM.named_parameters()]:
            for name, param in p:
                if 'weight' in name or 'bias' in name:
                    param.data.uniform_(-0.1, 0.1)

        for m in [self.query_linear]:
            # nn.init.xavier_normal_(m.weight)
            m.weight.data.uniform_(-0.1, 0.1)
            nn.init.constant_(m.bias, 0.)

    def get_lstm_packed_result(self, seqs, token_emb, dropout, lstm):
        seqs_len = (seqs != self.pad_id).sum(dim=-1)

        # batch_size = code_seqs.size(0)
        embs = token_emb(seqs)
        embs = dropout(embs)
        input_lens_sorted, indices = seqs_len.sort(descending=True)
        inputs_sorted = embs.index_select(0, indices)#按长度顺序重排input, [64, 6, 512]
        packed_code_embs = pack_padded_sequence(inputs_sorted, input_lens_sorted.data.tolist(), batch_first=True)#压缩
        
        # packed_code_embs = pack_padded_sequence(code_embs, lengths=(code_seqs != self.pad_id).sum(dim=-1).tolist(),
        #                                         enforce_sorted=False, batch_first=True)
        
        # h_n: 2, bs, hs
        hidden_states, (h_n, c_n) = lstm(packed_code_embs)

        _, inv_indices = indices.sort()
        h_n = h_n.index_select(1, inv_indices)

        # bs, 2*hs
        h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        return h_n

    def forward(self, query):

        query_repr = self.get_lstm_packed_result(query, self.query_emb, self.query_dropout_layer, self.query_LSTM)
        query_repr = self.query_linear(query_repr)
        query_repr = F.normalize(query_repr, p=2, dim=-1, eps=1e-5)
        return query_repr




class GraphEncoder(nn.Module):
    """
    MPNN from
    `Neural Message Passing for Quantum Chemistry <https://arxiv.org/abs/1704.01212>`__

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node feature, default to be 15.
    edge_input_dim : int
        Dimension of input edge feature, default to be 15.
    output_dim : int
        Dimension of prediction, default to be 12.
    node_hidden_dim : int
        Dimension of node feature in hidden layers, default to be 64.
    edge_hidden_dim : int
        Dimension of edge feature in hidden layers, default to be 128.
    num_step_message_passing : int
        Number of message passing steps, default to be 6.
    num_step_set2set : int
        Number of set2set steps
    num_layer_set2set : int
        Number of set2set layers
    """

    def __init__(
        self,
        positional_embedding_size=32,
        max_node_freq=8,
        max_edge_freq=8,
        max_degree=128,
        freq_embedding_size=32,
        degree_embedding_size=32,
        output_dim=32,
        node_hidden_dim=32,
        edge_hidden_dim=32,
        num_layers=6,
        num_heads=4,
        num_step_set2set=6,
        num_layer_set2set=3,
        norm=False,
        gnn_model="mpnn",
        degree_input=False,
        lstm_as_gate=False,
    ):
        super(GraphEncoder, self).__init__()

        if degree_input:
            node_input_dim = positional_embedding_size + degree_embedding_size
        else:
            node_input_dim = positional_embedding_size
        # node_input_dim = (
        #     positional_embedding_size + freq_embedding_size + degree_embedding_size + 3
        # )
        edge_input_dim = freq_embedding_size + 1
        if gnn_model == "mpnn":
            self.gnn = UnsupervisedMPNN(
                output_dim=output_dim,
                node_input_dim=node_input_dim,
                node_hidden_dim=node_hidden_dim,
                edge_input_dim=edge_input_dim,
                edge_hidden_dim=edge_hidden_dim,
                num_step_message_passing=num_layers,
                lstm_as_gate=lstm_as_gate,
            )
        elif gnn_model == "gat":
            self.gnn = UnsupervisedGAT(
                node_input_dim=node_input_dim,
                node_hidden_dim=node_hidden_dim,
                edge_input_dim=edge_input_dim,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        elif gnn_model == "gin":
            self.gnn = UnsupervisedGIN(
                num_layers=num_layers,
                num_mlp_layers=2,
                input_dim=node_input_dim,
                hidden_dim=node_hidden_dim,
                output_dim=output_dim,
                final_dropout=0.5,
                learn_eps=False,
                graph_pooling_type="sum",
                neighbor_pooling_type="sum",
                use_selayer=False,
            )
        self.gnn_model = gnn_model

        self.max_node_freq = max_node_freq
        self.max_edge_freq = max_edge_freq
        self.max_degree = max_degree
        self.degree_input = degree_input

        # self.node_freq_embedding = nn.Embedding(
        #     num_embeddings=max_node_freq + 1, embedding_dim=freq_embedding_size
        # )
        if degree_input:
            self.degree_embedding = nn.Embedding(
                num_embeddings=max_degree + 1, embedding_dim=degree_embedding_size
            )

        # self.edge_freq_embedding = nn.Embedding(
        #     num_embeddings=max_edge_freq + 1, embedding_dim=freq_embedding_size
        # )

        self.set2set = Set2Set(node_hidden_dim, num_step_set2set, num_layer_set2set)
        self.lin_readout = nn.Sequential(
            nn.Linear(2 * node_hidden_dim, node_hidden_dim),
            nn.ReLU(),
            nn.Linear(node_hidden_dim, output_dim),
        )
        self.norm = norm

    def forward(self, g, return_all_outputs=False):
        """Predict molecule labels

        Parameters
        ----------
        g : DGLGraph
            Input DGLGraph for molecule(s)
        n_feat : tensor of dtype float32 and shape (B1, D1)
            Node features. B1 for number of nodes and D1 for
            the node feature size.
        e_feat : tensor of dtype float32 and shape (B2, D2)
            Edge features. B2 for number of edges and D2 for
            the edge feature size.

        Returns
        -------
        res : Predicted labels
        """

        # nfreq = g.ndata["nfreq"]
        if self.degree_input:
            device = g.ndata["pos_undirected"].device
            degrees = g.in_degrees()
            if device != torch.device("cpu"):
                degrees = degrees.cuda(device)

            n_feat = torch.cat(
                (
                    g.ndata["pos_undirected"],
                    self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    # g.ndata["seed"].unsqueeze(1).float(),
                ),
                dim=-1,
            )
        else:
            n_feat = torch.cat(
                (
                    g.ndata["pos_undirected"],
                    # g.ndata["pos_directed"],
                    # self.node_freq_embedding(nfreq.clamp(0, self.max_node_freq)),
                    # self.degree_embedding(degrees.clamp(0, self.max_degree)),
                    g.ndata["seed"].unsqueeze(1).float(),
                    # nfreq.unsqueeze(1).float() / self.max_node_freq,
                    # degrees.unsqueeze(1).float() / self.max_degree,
                ),
                dim=-1,
            )

        # efreq = g.edata["efreq"]
        # e_feat = torch.cat(
        #     (
        #         self.edge_freq_embedding(efreq.clamp(0, self.max_edge_freq)),
        #         efreq.unsqueeze(1).float() / self.max_edge_freq,
        #     ),
        #     dim=-1,
        # )
        e_feat = None
        if self.gnn_model == "gin":
            x, all_outputs = self.gnn(g, n_feat, e_feat)
        else:
            x, all_outputs = self.gnn(g, n_feat, e_feat), None
            x = self.set2set(g, x)
            x = self.lin_readout(x)
        if self.norm:
            x = F.normalize(x, p=2, dim=-1, eps=1e-5)
        if return_all_outputs:
            return x, all_outputs
        else:
            return x


if __name__ == "__main__":
    model = GraphEncoder(gnn_model="gin")
    print(model)
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 0, 1, 2], [1, 2, 2, 1])
    g.ndata["pos_directed"] = torch.rand(3, 16)
    g.ndata["pos_undirected"] = torch.rand(3, 16)
    g.ndata["seed"] = torch.zeros(3, dtype=torch.long)
    g.ndata["nfreq"] = torch.ones(3, dtype=torch.long)
    g.edata["efreq"] = torch.ones(4, dtype=torch.long)
    g = dgl.batch([g, g, g])
    y = model(g)
    print(y.shape)
    print(y)
