import json
from tqdm import tqdm
from tree_sitter import Language, Parser
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from vocab import WordVocab
import re
import pickle
import os
import ast
import torch as th
import dgl
from dgl.data.utils import save_graphs, load_graphs
import datetime
import astunparse
import random

def split_camel(camel_str, test=''):
    try:
        split_str = re.sub(
            r'(?<=[a-z]|[0-9])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+',
            '_',
            camel_str)
    except TypeError:
        return ['']
    try:
        if split_str[0] == '_':
            return [camel_str]
    except IndexError:
        return []
    return [i for i in split_str.lower().split('_') if i != '']

def get_nodes_level(code):
    tree = ast.parse(code).body[0]
    tokens = []
    for node in ast.walk(tree):
        tokens.append(type(node).__name__)
    return tokens


def get_ast_seq_level(tree):
    astnodes = []
    nodes = []
    current_idx = 0
    for node in ast.walk(tree):
        # 因为load，store这种可能会重复出现，所以加个判断，重复出现就不重新加id了
        if node not in nodes:
            nodes.append(node)
            node.idx = current_idx
            current_idx += 1
        # 加上父节点
        if not next(ast.iter_child_nodes(node), None) is None:
            for child in ast.iter_child_nodes(node):
                child.parent = node

        astnodes.append(type(node).__name__)
    return astnodes


def get_vars(tree):
    identifiers = []
    for node in ast.walk(tree):
        # 获取部分变量，注意这里应该不是全部的变量,获取全部变量还是有点麻烦
        if type(node).__name__ == 'Name' and type(node.parent).__name__ != 'Call' and node.id != 'self':
            identifiers.append(node.id)
        if type(node).__name__ == 'arg' and type(node.parent.parent).__name__ == 'FunctionDef' and node.arg != 'self':
            identifiers.append(node.arg)
    return list(set(identifiers))

def tokenizer(vocab, seqs, output=None):
    for seq in seqs:
        for i in range(len(seq)):
            # 如果是列表的话，就要再来一个for循环
            if isinstance(seq[i], list):
                for j in range(len(seq[i])):
                    seq[i][j] = vocab.stoi.get(seq[i][j], vocab.unk_index)
            else:
                seq[i] = vocab.stoi.get(seq[i], vocab.unk_index)
    if output:
        f1 = open(output, 'wb')
        pickle.dump(seqs, f1)
    return seqs


def get_ast_graph(tree, ast_vocab):
    u = []
    v = []
    node_type = []
    astnodes = []
    for node in ast.walk(tree):
        if node not in astnodes:
            astnodes.append(node)
            node_type.append(ast_vocab.stoi.get(
                type(node).__name__, ast_vocab.unk_index))
        if not next(ast.iter_child_nodes(node), None) is None:
            for child in ast.iter_child_nodes(node):
                u.append(int(node.idx))
                v.append(int(child.idx))
    node_type = th.tensor(node_type)
    u = th.tensor(u)
    v = th.tensor(v)
    graph = dgl.graph((u, v))
    # 无向图
    graph = dgl.to_bidirected(graph)
    graph.ndata['node_type'] = node_type
    return graph


def get_token_vocab(data_path):
    train_data_path = os.path.join(data_path, 'csn_train.json')
    query_vocab_path = os.path.join(data_path, 'query_vocab.pickle')
    ast_vocab_path = os.path.join(data_path, 'ast_vocab.pickle')
    if os.path.exists(query_vocab_path) and os.path.exists(ast_vocab_path):
        query_vocab = pickle.load(open(query_vocab_path, 'rb'))
        ast_vocab = pickle.load(open(ast_vocab_path, 'rb'))
    else:
        vocab_query = []
        vocab_ast = []
        f = open(train_data_path)
        js = json.load(f)

        for line in tqdm(js):
            code = line['code']
            # code_tokens = line['code_tokens']
            query = line['doc']

            query = query.replace('\n', ' ')  # 去掉回车
            query = re.sub(
                r'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>', '', query)  # 去掉括号内容
            query = query.replace('_', ' ').strip()
            query = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip()
            query_tokens = split_camel(query)  # 拆驼峰命名

            ast_nodes = get_nodes_level(code)

            if not query_tokens or not ast_nodes:
                print(1234)

            vocab_query += query_tokens
            vocab_ast += ast_nodes

        query_vocab = WordVocab(vocab_query, max_size=50000,
                                min_freq=1)
        query_vocab.save_vocab(query_vocab_path)

        ast_vocab = WordVocab(vocab_ast, max_size=10000,
                              min_freq=1)
        ast_vocab.save_vocab(ast_vocab_path)

    return query_vocab, ast_vocab


def get_train_data(query_vocab, ast_vocab, data_path):
    train_data_path = os.path.join(data_path, 'csn_train.json')
    train_query_path = os.path.join(data_path, 'query_dataset_train.pickle')
    var_vocab_path = os.path.join(data_path, 'var_vocab.pickle')
    train_ast_graph_path = os.path.join(data_path, 'graph_data_train_nodetype.pickle')
    codes_path = os.path.join(data_path, 'train_codes.pickle')
    if os.path.exists(train_query_path) and os.path.exists(var_vocab_path) and os.path.exists(train_ast_graph_path) and os.path.exists(codes_path):
        query_seq_list = pickle.load(open(train_query_path, 'rb'))
        var_vocab = pickle.load(open(var_vocab_path, 'rb'))
        ast_graph_list, _ = load_graphs(train_ast_graph_path)
        codes = pickle.load(open(codes_path, 'rb'))
    else:
        f = open(train_data_path)
        lines = json.load(f)
        query_seq_list = []
        code_seq_list = []
        codes = []
        var_vocab = []
        ast_seq_list = []
        ast_seq_level_list = []
        ast_graph_list = []
        # 从根节点到所有叶子节点的路径
        ast_path_list = []

        for js in tqdm(lines):
            query = js['doc']
            query = query.replace('\n', ' ')  # 去掉回车
            query = re.sub(
                r'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>', '', query)  # 去掉括号内容
            query = query.replace('_', ' ').strip()
            query = re.sub(r'[^a-zA-Z0-9 ]', '', query).strip()
            query_seq_list.append(split_camel(query))  # 拆驼峰命名

            code = js['code']
            codes.append(code)
            tree = ast.parse(code).body[0]
            
            # 标记下id和parent
            get_ast_seq_level(tree)

            var_vocab += get_vars(tree)

            ast_graph_list.append(get_ast_graph(tree, ast_vocab))

        query_seq_list = tokenizer(
            query_vocab, query_seq_list, train_query_path)

        save_graphs(train_ast_graph_path, ast_graph_list)

        pickle.dump(list(set(var_vocab)), open(var_vocab_path, 'wb'))
        pickle.dump(codes, open(codes_path, 'wb'))
        
    return query_seq_list, codes, var_vocab, ast_graph_list
