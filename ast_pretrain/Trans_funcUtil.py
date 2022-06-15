from hashlib import new
import json
from sklearn.metrics import jaccard_score
from torch import rand
from tqdm import tqdm
from parser import (remove_comments_and_docstrings,
                    tree_to_token_index,
                    index_to_code_token,
                    tree_to_variable_index)
from tree_sitter import Language, Parser
from queue import Queue
from re import T
import json
from tqdm import tqdm
import json
from tqdm import tqdm
import random
import ast
import datetime
import astunparse
# python java ruby go php javascript c-sharp
lang = 'python'

def pre_traversal(node, nodes, current_idx):
    if node not in nodes:
        node.idx = current_idx
        current_idx += 1
    nodes.append(node)

    if not next(ast.iter_child_nodes(node), None) is None:
        for child in ast.iter_child_nodes(node):
            child.parent = node
            current_idx = pre_traversal(child, nodes, current_idx)
    return current_idx


def get_vars(code):
    tree = ast.parse(code)
    tree = tree.body[0]
    identifiers = []
    nodes  = []
    pre_traversal(tree, nodes, 0)
    for node in nodes:
        # 获取部分变量，注意这里应该不是全部的变量,获取全部变量还是有点麻烦
        if type(node).__name__ == 'Name' and type(node.parent).__name__ != 'Call':
            identifiers.append(node.id)
        if type(node).__name__ == 'arg' and type(node.parent.parent).__name__ == 'FunctionDef' and node.arg != 'self':
            identifiers.append(node.arg)
    return list(set(identifiers))

# 这个获取的内容比get_vars多，为什么不全用这个呢？因为这个获取的变量虽然更多，但是不全是能改的，但是上面那个获取的虽然不多，但都是能改的
def get_vars_by_node(tree):
    identifiers = []
    nodes  = []
    pre_traversal(tree, nodes, 0)
    for node in nodes:
        # 如何去识别出所有的变量是个问题，万一漏了一个就很有可能会导致换行失败
        if type(node).__name__ == 'Global':
            identifiers.extend(node.names)
        if type(node).__name__ == 'Name':
            identifiers.append(node.id)
        if type(node).__name__ == 'alias':
            if hasattr(node, 'asname'):
                identifiers.append(node.asname)
            identifiers.append(node.name)
    return list(set(identifiers))



def isdepend(depend, offer):
    for d in depend:
        if d in offer:
            return True
    return False

def permulate_statement(code, var_vocab, identifiers):
    # code = b
    tree = ast.parse(code)
    tree = tree.body[0]
    nodes = []
    pre_traversal(tree, nodes, 0)

    for node in nodes:
        # 后者是排除lambada
        if hasattr(node, 'body') and isinstance(node.body, list):
            # 看他body里的语句，而不是用ast.iter_child_nodes(node), 后者会把if和else里的内容放在一起
            childs = node.body
            # 对哪些变量进行了操作
            variables = []

            for child in childs:
                variables.append(get_vars_by_node(child))

            matrix = [[1 for i in range(len(variables))] for j in range(len(variables))] 
            replace = []
            for i in range(1, len(childs)):
                truncate_node_type = ['Return','Raise','Yield','Continue','Break', 'Pass']
                
                # 如果有return，或者有sys.exit(), 或者有raise，或者有field，continue,break,pass,就直接不往下看了，理论上下面也没内容了，就算有也没有意义
                if type(childs[i]).__name__ in truncate_node_type:
                    break
                truncate_node_str = ['exit', 'quit', 'os.', 'run(', 'sys.']
                # 和os相关的全杀了，宁错杀，不放过, 加.,加(是尽可能区分一下，尽量少错杀一点
                # 保险起见，像有quit啊，exit这种的我都像return那样处理了吧，主要我也不知道他们在树结构上有什么特点，所以只能转成代码检查了,
                node_code = astunparse.unparse(ast.fix_missing_locations(childs[i]))
                sign = True
                for s in truncate_node_str:
                    if s in node_code:
                        sign = False
                        break
                if not sign:
                    break 

                for j in range(i-1,-1,-1):
                    # 由依赖传递产生的
                    if matrix[i][j] == 0:
                        continue
                    # 如果第i个句子依赖于第j个句子，那么前j个句子都不能和i交换
                    if isdepend(variables[i], variables[j]):
                        matrix[i][:j + 1] = [0 for i in range(j + 1)]
                        # matrix[i][:j] = matrix[j][:j]
                    else:
                        replace.append((i,j))
            if replace:
                r = random.sample(replace,1)[0]
                node.body[r[0]], node.body[r[1]] = node.body[r[1]], node.body[r[0]]
    
    new_code = astunparse.unparse(ast.fix_missing_locations(tree))
    return new_code

def loop_exchange(code, var_vocab, identifiers):
    # code = a 
    tree = ast.parse(code)
    tree = tree.body[0]
    nodes = []
    variables = []
    expr = []
    ranges = []
    pre_traversal(tree, nodes, 0)
    
    # 因为改一个for以后，tree就变了，你就没法继续这个循环了，而如果你在外面再套一个while，终止条件又不太好写
    # 所以干脆不管你代码里有多少for，我只改一个for，我把nodes这个列表shuffle一下，就能保证不是每次都值逮着第一个for在改
    random.shuffle(nodes)
    for node in nodes:
        if type(node).__name__ == 'For':
            childs = list(ast.iter_child_nodes(node))

            # 开头第一个孩子是in前面的， 第二个孩子是in后面的
            variables = childs[:2]
            # 剩下的就是expr
            expr = childs[2:]

            # 构造初始值, 假设只有一个递增的数
            candidate = 'ijklmnopqrst'
            for i in candidate:
                if i not in identifiers:
                    identifiers.append(i)
                    break

            # 从0开始
            s = i + " = 0"                
            assign_tree = ast.parse(s)
            # 插入到for的前一行
            if node in node.parent.body:
                node.parent.body.insert(node.parent.body.index(node), assign_tree)     
            elif node in node.parent.orelse:
                node.parent.body.insert(node.parent.orelse.index(node), assign_tree) 

            # variables中第二个可能是一个迭代器，要将其转换为list
            for j in candidate:
                if j not in identifiers:
                    identifiers.append(j)
                    break
            s = j + " = list(10)"
            assign_tree = ast.parse(s)
            assign_tree.body[0].value.args = [variables[1]]
            if node in node.parent.body:
                node.parent.body.insert(node.parent.body.index(node), assign_tree)     
            elif node in node.parent.orelse:
                node.parent.body.insert(node.parent.orelse.index(node), assign_tree) 

            # 自增
            s = i + " += 1"
            augassign_tree = ast.parse(s)
            expr.insert(0, augassign_tree)

            # 还要写一个variables[0] = variables[1][i]放在while下的开头
            s = "a = " + j + '[' + i + ']'
            assign_tree = ast.parse(s)
            assign_tree.body[0].targets = [variables[0]]
            expr.insert(0, assign_tree)

            # 构造while
            s = "while " + i + " < len(" + j + ")" + ":\n    print()"
            while_tree = ast.parse(s)
            while_tree.body[0].body = expr
            if node in node.parent.body:
                node.parent.body[node.parent.body.index(node)] = while_tree  
            elif node in node.parent.orelse:
                node.parent.orelse[node.parent.orelse.index(node)] = while_tree
            break
    
    new_code = astunparse.unparse(ast.fix_missing_locations(tree))
    return new_code
            

def get_ust(var_vocab, variables, identifiers):
    unused_str = ''
    # a = b
    ran = random.random()
    if  ran < 0.4:
        new_variables = random.sample(var_vocab, 1)[0]
        while new_variables in identifiers:
            new_variables = random.sample(var_vocab, 1)[0]
        unused_str += new_variables + " = " + random.sample(variables, 1)[0]
    elif ran < 0.7:
        unused_str += "print(" + random.sample(variables, 1)[0]
        while random.random() < 0.2:
            unused_str += ", " + random.sample(variables, 1)[0]
        unused_str += ")"
    else:
        unused_str += 'if 1:\n    print("' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '")'
    unused_tree = ast.parse(unused_str)
    return unused_tree

def unused_statement(code, var_vocab, identifiers):
    tree = ast.parse(code)
    tree = tree.body[0]
    nodes  = []
    pre_traversal(tree, nodes, 0)
    # 当前可用的变量，也不是全部变量
    variables = ['0']
    count = 0
    for node in nodes:
        # child_list = list(ast.iter_child_nodes(node))
        
        if type(node).__name__ == 'Name' and ('Assign' in type(node.parent).__name__  or type(node.parent).__name__ == 'withitem'):
            variables.append(node.id)
        if type(node).__name__ == 'arg' and type(node.parent.parent).__name__ == 'FunctionDef' and node.arg != 'self':
            variables.append(node.arg)

        # 感觉好像只要有body这个属性，就可以插入, 后者是排除lambada
        if hasattr(node, 'body') and isinstance(node.body, list):
            if random.random() < 0.6:
                count += 1
                node.body.insert(random.randint(0, len(node.body) - 1), get_ust(var_vocab, variables, identifiers))
    
    if count == 0:
        tree = ast.parse(code)
        tree = tree.body[0]
        unused_str = "timestamp = '" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "'"
        unused_tree = ast.parse(unused_str)
        if hasattr(tree, 'body') and isinstance(tree.body, list):
            tree.body.insert(random.randint(0, len(tree.body) - 1), unused_tree)
    new_code = astunparse.unparse(ast.fix_missing_locations(tree))
    return new_code

def variable_rename(code, var_vocab, identifiers):
    tree = ast.parse(code)
    tree = tree.body[0]
    nodes  = []
    pre_traversal(tree, nodes, 0)
    variables = {}

    for node in nodes:
        # child_list = list(ast.iter_child_nodes(node))
        # 记录哪些变量要改
        if type(node).__name__ == 'Name' and ('Assign' in type(node.parent).__name__  or type(node.parent).__name__ == 'withitem'):
            new_var = random.sample(var_vocab, 1)[0]
            while new_var in identifiers or new_var in variables:
                new_var = random.sample(var_vocab, 1)[0]
            if node.id not in variables:
                variables[node.id] = new_var
        if type(node).__name__ == 'arg' and type(node.parent.parent).__name__ == 'FunctionDef' and node.arg != 'self':
            new_var = random.sample(var_vocab, 1)[0]
            while new_var in identifiers or new_var in variables:
                new_var = random.sample(var_vocab, 1)[0]
            if node.arg not in variables:
                variables[node.arg] = new_var      

    for node in nodes:
        # child_list = list(ast.iter_child_nodes(node))
        if type(node).__name__ == 'Name':
            if node.id in variables:
                node.id = variables[node.id]

        if type(node).__name__ == 'arg':
            if node.arg in variables:
                node.arg = variables[node.arg]

    new_code = astunparse.unparse(ast.fix_missing_locations(tree))
    return new_code




def invokeTransformations(code, var_vocab):
    identifiers = get_vars(code)
    trans_funcs = [permulate_statement, loop_exchange, unused_statement, variable_rename]

    use_func = random.sample(trans_funcs, 2)

    for func in use_func:
        code = func(code, var_vocab, identifiers)
    
    return code



