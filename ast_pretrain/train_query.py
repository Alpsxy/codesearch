import argparse
import copy
import os
import time
import warnings

import dgl
import numpy as np
import psutil
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from criterions import NCESoftmaxLoss, NCESoftmaxLossNS
from memory_moco import MemoryMoCo
from graph_dataset import (
    LoadBalanceGraphDataset,
    worker_init_fn,
)

from models import GraphEncoder, QueryEncoder
import random
import warnings
warnings.filterwarnings("ignore")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x / warmup
    return max((x - 1.0) / (warmup - 1.0), 0)


def set_seed(seed=45):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def parse_option():

    # fmt: off
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print-freq", type=int, default=10, help="print frequency")
    parser.add_argument("--tb-freq", type=int, default=250, help="tb frequency")
    parser.add_argument("--save-freq", type=int, default=1, help="save frequency")
    parser.add_argument("--batch-size", type=int, default=256, help="batch_size")
    parser.add_argument("--num-workers", type=int, default=12, help="num of workers to use")
    parser.add_argument("--num-copies", type=int, default=2, help="num of dataset copies that fit in memory")
    parser.add_argument("--num-samples", type=int, default=8000, help="num of samples per batch per worker")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")

    # optimization
    parser.add_argument("--optimizer", type=str, default='adam', choices=['sgd', 'adam', 'adagrad'], help="optimizer")

    parser.add_argument("--lr_decay_epochs", type=str, default="120,160,200", help="where to decay lr, can be a list")
    parser.add_argument("--lr_decay_rate", type=float, default=0.0, help="decay rate for learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="clip norm")

    # resume
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")

    # augmentation setting
    parser.add_argument("--aug", type=str, default="1st", choices=["1st", "2nd", "all"])

    parser.add_argument("--exp", type=str, default="Pretrain")

    # dataset definition
    parser.add_argument("--dataset", type=str, default="dgl", choices=["dgl", "wikipedia", "blogcatalog", "usa_airport", "brazil_airport", "europe_airport", "cora", "citeseer", "pubmed", "kdd", "icdm", "sigir", "cikm", "sigmod", "icde", "h-index-rand-1", "h-index-top-1", "h-index"])

    # model definition
    parser.add_argument("--model", type=str, default="gin", choices=["gat", "mpnn", "gin"])
    # other possible choices: ggnn, mpnn, graphsage ...
    parser.add_argument("--num-layer", type=int, default=5, help="gnn layers")
    parser.add_argument("--readout", type=str, default="avg", choices=["avg", "set2set"])
    parser.add_argument("--set2set-lstm-layer", type=int, default=3, help="lstm layers for s2s")
    parser.add_argument("--set2set-iter", type=int, default=6, help="s2s iteration")
    parser.add_argument("--norm", action="store_true", default=True, help="apply 2-norm on output feats")

    parser.add_argument("--query_emb_size", type=int, default=128, choices=["avg", "set2set"])
    parser.add_argument("--query_lstm_size", type=int, default=256, help="lstm layers for s2s")
    parser.add_argument("--query_hidden_size", type=int, default=128, help="s2s iteration")



    # loss function
    parser.add_argument("--nce-k", type=int, default=32)
    parser.add_argument("--nce-t", type=float, default=0.07)

    # random walk
    parser.add_argument("--rw-hops", type=int, default=256)
    parser.add_argument("--subgraph-size", type=int, default=128)
    parser.add_argument("--restart-prob", type=float, default=0.2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--positional-embedding-size", type=int, default=32)
    parser.add_argument("--max-node-freq", type=int, default=16)
    parser.add_argument("--max-edge-freq", type=int, default=16)
    parser.add_argument("--max-degree", type=int, default=512)
    parser.add_argument("--freq-embedding-size", type=int, default=16)
    parser.add_argument("--degree-embedding-size", type=int, default=16)

    parser.add_argument("--max_query_len", type=int, default=100)
    parser.add_argument("--pad_id", type=int, default=0)



    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")

    # specify folder
    parser.add_argument("--model-path", type=str, default='ast_pretrain/saved_models/GCC_CSN_query_lr0.001', help="path to save model")
    parser.add_argument("--data-path", type=str, default='data/pretrain', help="path to save model")

    # GPU setting
    parser.add_argument("--gpu", default=[2], type=int, nargs='+', help="GPU id to use.")

    # memory setting
    parser.add_argument("--moco", action="store_true", help="using MoCo (otherwise Instance Discrimination)")

    parser.add_argument("--alpha", type=float, default=0.999, help="exponential moving average weight")


    # cross validation
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--fold-idx", type=int, default=0, help="random seed.")
    parser.add_argument("--cv", action="store_true")
    # fmt: on

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(",")
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr


def batcher():
    def batcher_dev(batch):
        query, graph_q, graph_k = zip(*batch)
        #batch的效果是将 n 张小图打包在一起的操作可以看成是生成一张含 n 个不相连小图的大图。
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return torch.tensor(query), graph_q, graph_k

    return batcher_dev

def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1 - m, p1.detach().data)

def clip_grad_norm(params, max_norm):
    """Clips gradient norm."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        return torch.sqrt(
            sum(p.grad.data.norm() ** 2 for p in params if p.grad is not None)
        )

def train_moco(
    epoch, train_loader, model_query, model, model_ema, contrast, criterion, optimizer, args
):

    n_batch = train_loader.dataset.total // args.batch_size
    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.train()

    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    gnorm_meter = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        query, graph_q, graph_k = batch
        query = query.to(torch.device(args.gpu))
        graph_q = graph_q.to(torch.device(args.gpu))
        graph_k = graph_k.to(torch.device(args.gpu))
 
        bsz = graph_q.batch_size

        if args.moco:
            # ===================Moco forward=====================
            feat_q = model(graph_q)
            with torch.no_grad():
                feat_k = model_ema(graph_k)

            out = contrast(feat_q, feat_k)
            prob = out[:, 0].mean()
        else:
            # ===================Negative sampling forward=====================
            # 2 * 64
            feat_q = model(graph_q)
            feat_k = model(graph_k)

            feat_query = model_query(query)

            sim1 = torch.matmul(feat_query, feat_q.t()) / args.nce_t
            sim2 = torch.matmul(feat_query, feat_k.t()) / args.nce_t
            sim3 = torch.matmul(feat_k, feat_q.t()) / args.nce_t

        optimizer.zero_grad()
        # loss = criterion(sim3)
        loss = criterion(sim1) + criterion(sim2) + criterion(sim3)
        loss.backward()
        # 梯度裁剪
        grad_norm = clip_grad_norm(model.parameters(), args.clip_norm)

        global_step = epoch * n_batch + idx

        lr_this_step = args.learning_rate * warmup_linear(
            global_step / (args.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        graph_size.update(
            (graph_q.number_of_nodes() + graph_k.number_of_nodes()) / 2.0 / bsz, 2 * bsz
        )
        gnorm_meter.update(grad_norm, 1)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        if args.moco:
            moment_update(model, model_ema, args.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % args.print_freq == 0:
            mem = psutil.virtual_memory()
            #  print(f'{idx:8} - {mem.percent:5} - {mem.free/1024**3:10.2f} - {mem.available/1024**3:10.2f} - {mem.used/1024**3:10.2f}')
            #  mem_used.append(mem.used/1024**3)
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "prob {prob.val:.3f} ({prob.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    prob=prob_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )

# def main(args, trial):
def main(args):
    for key, value in vars(args).items():
        print("[setting] " + key + ": " + str(value))
    assert args.gpu is not None and torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))
    assert args.positional_embedding_size % 2 == 0
    print("setting random seeds")
    # 可用内存
    mem = psutil.virtual_memory()
    print("before construct dataset", mem.used / 1024 ** 3)    

    train_dataset = LoadBalanceGraphDataset(
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        data_path=args.data_path,#./data/small.bin  /data1/jdfeng/code_ast/graph_data/python_graph_small.bin
        num_copies=args.num_copies,
        positional_embedding_size=args.positional_embedding_size,
        max_query_len=args.max_query_len,
        pad_id=args.pad_id
    )
    mem = psutil.virtual_memory()
    print("before construct dataloader", mem.used / 1024 ** 3)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    n_data = None
    
    model_query = QueryEncoder(len(train_dataset.query_vocab), args.query_emb_size, args.query_lstm_size, args.query_hidden_size)

    model, model_ema = [
        GraphEncoder(
            positional_embedding_size=args.positional_embedding_size,
            max_node_freq=args.max_node_freq,
            max_edge_freq=args.max_edge_freq,
            max_degree=args.max_degree,
            freq_embedding_size=args.freq_embedding_size,
            degree_embedding_size=args.degree_embedding_size,
            output_dim=args.hidden_size,
            node_hidden_dim=args.hidden_size,
            edge_hidden_dim=args.hidden_size,
            num_layers=args.num_layer,
            num_step_set2set=args.set2set_iter,
            num_layer_set2set=args.set2set_lstm_layer,
            norm=args.norm,
            gnn_model=args.model,
            degree_input=True,
        )
        for _ in range(2)
    ]

    # copy weights from `model' to `model_ema'
    if args.moco:
        moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(
        args.hidden_size, n_data, args.nce_k, args.nce_t, use_softmax=True
    ).cuda(args.gpu)

    criterion = NCESoftmaxLoss() if args.moco else NCESoftmaxLossNS()
    criterion = criterion.cuda(args.gpu)

    model_query = model_query.cuda(args.gpu)
    model = model.cuda(args.gpu)
    model_ema = model_ema.cuda(args.gpu)


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    args.start_epoch = 1

    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")
        time1 = time.time()
        loss = train_moco(
            epoch,
            train_loader,
            model_query, 
            model,
            model_ema,
            contrast,
            criterion,
            optimizer,
            args,
        )        
        time2 = time.time()
        print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

        # save model
        if epoch % args.save_freq == 0:
            print("==> Saving...")
            state = {
                "opt": args,
                "model": model.state_dict(),
                "model_query": model_query.state_dict(),
                "contrast": contrast.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            if args.moco:
                state["model_ema"] = model_ema.state_dict()
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)       
            save_file = os.path.join(
                args.model_path, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            torch.save(state, save_file)
            # help release GPU memory
            del state
            torch.cuda.empty_cache()



if __name__ == "__main__":
    
    args = parse_option()
    set_seed(args.seed)
    args.gpu = args.gpu[0]
    main(args)