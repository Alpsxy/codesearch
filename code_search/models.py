import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from transformers.modeling_bert import BertLayerNorm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaModel, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from transformers.modeling_utils import PreTrainedModel
import math

class ModelBinary(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelBinary, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        # logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 1))
        logits = self.mlp(torch.cat((nl_vec, code_vec), 1))
        loss = self.loss_func(logits, labels.float())
        predictions = (logits > 0.5).int()  # (Batch, )
        return loss, predictions


class ModelContra(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args, ast_encoder=None):
        super(ModelContra, self).__init__(config)
        self.encoder = encoder
        self.ast_encoder = ast_encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
                                 
        self.loss_func = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]

        code_vec = F.normalize(code_vec, p=2, dim=-1, eps=1e-5)
        nl_vec = F.normalize(nl_vec, p=2, dim=-1, eps=1e-5)

        sims1 = torch.matmul(nl_vec, code_vec.t()) / 0.07
        sims2 = torch.matmul(code_vec, nl_vec.t()) / 0.07


        sims1 = sims1[labels == 1]
        sims2 = sims2[labels == 1]

        pos_size = sims1.shape[0]

        label = torch.nonzero(labels==1).squeeze()

        loss = (self.loss_func(sims1, label) + self.loss_func(sims2, label)) / pos_size

        if return_vec:
            return code_vec, nl_vec
        else:
            return loss

# 加入astseq
class ModelContraASTSeq(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args, ast_encoder=None):
        super(ModelContraASTSeq, self).__init__(config)
        self.encoder = encoder
        self.ast_encoder = ast_encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())

        self.query_linear = nn.Linear(768, 768)
        self.code_linear = nn.Linear(768 + 512, 768)

        self.loss_func = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, ast_seq, ast_seq_level, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]

        ast_vec = self.ast_encoder(ast_seq, ast_seq_level)

        code_vec = torch.cat((code_vec, ast_vec), -1)

        code_vec = self.code_linear(code_vec)
        nl_vec = self.query_linear(nl_vec)

        code_vec = F.normalize(code_vec, p=2, dim=-1, eps=1e-5)
        nl_vec = F.normalize(nl_vec, p=2, dim=-1, eps=1e-5)

        sims1 = torch.matmul(nl_vec, code_vec.t()) / 0.07
        sims2 = torch.matmul(code_vec, nl_vec.t()) / 0.07


        sims1 = sims1[labels == 1]
        sims2 = sims2[labels == 1]

        pos_size = sims1.shape[0]

        label = torch.nonzero(labels==1).squeeze()

        loss = (self.loss_func(sims1, label) + self.loss_func(sims2, label)) / pos_size

        if return_vec:
            return code_vec, nl_vec
        else:
            return loss


    # def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
    #     bs = code_inputs.shape[0]
    #     inputs = torch.cat((code_inputs, nl_inputs), 0)
    #     outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
    #     code_vec = outputs[:bs]
    #     nl_vec = outputs[bs:]

    #     code_vec = F.normalize(code_vec, p=2, dim=-1, eps=1e-5)
    #     nl_vec = F.normalize(nl_vec, p=2, dim=-1, eps=1e-5)

    #     label = torch.arange(bs).cuda().long()

    #     sims = torch.matmul(nl_vec, code_vec.t()) / 0.07

    #     loss = self.loss_func(sims, label)

    #     if return_vec:
    #         return code_vec, nl_vec
    #     else:
    #         return loss



class ModelContraOnline(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelContraOnline, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        nl_vec = nl_vec.unsqueeze(1).repeat([1, bs, 1])
        code_vec = code_vec.unsqueeze(0).repeat([bs, 1, 1])
        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 2)).squeeze(2) # (Batch, Batch)
        matrix_labels = torch.diag(labels).float()  # (Batch, Batch)
        poss = logits[matrix_labels==1]
        negs = logits[matrix_labels==0]

        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        loss = - (torch.log(1 - negative_pairs).mean() + torch.log(positive_pairs).mean())
        predictions = (logits.gather(0, torch.arange(bs, device=loss.device).unsqueeze(0)).squeeze(0) > 0.5).int()
        return loss, predictions

