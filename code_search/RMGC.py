from turtle import forward
import torch
from torch import dropout, nn
from torch.nn.functional import cosine_similarity
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class RMGC(nn.Module):

    def __init__(self):
        super(RMGC, self).__init__()
        ast_vocab_size = 101
        self.pad_id = 0
        self.dropout = 0.25
        self.ast_emb_size = 32
        self.ast_lstm_size = 128

        # 所有astnode用同一个Embedding
        self.ast_emb = nn.Embedding(num_embeddings=ast_vocab_size, embedding_dim=self.ast_emb_size, padding_idx=self.pad_id)
        self.ast_seq_dropout_layer = nn.Dropout(self.dropout)

        # 相互学习部分要准备两个astseq的编码器，分别表示先序遍历的编码和层序遍历的表示。
        self.ast_seq_pre_LSTM = nn.LSTM(self.ast_emb_size, self.ast_lstm_size, batch_first=True, bidirectional=True)
        self.ast_seq_level_LSTM = nn.LSTM(self.ast_emb_size, self.ast_lstm_size, batch_first=True, bidirectional=True)


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



    def  get_ast_repr(self, ast_seq, ast_seq_level):
        ast_seq_pre_repr = self.get_lstm_packed_result(ast_seq, self.ast_emb, self.ast_seq_dropout_layer, self.ast_seq_pre_LSTM)
        ast_seq_level_repr = self.get_lstm_packed_result(ast_seq_level, self.ast_emb, self.ast_seq_dropout_layer, self.ast_seq_level_LSTM)
        # code_repr = self.fusion_linear(code_seq_repr)
        return torch.cat([ast_seq_pre_repr, ast_seq_level_repr], dim=-1)

    def forward(self, ast_seq, ast_seq_level):
        return self.get_ast_repr(ast_seq, ast_seq_level)