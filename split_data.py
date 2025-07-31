import sys
sys.argv = ['run.py']
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from time import time
from prettytable import PrettyTable
import logging
from utils.parser import parse_args
import dgl
import dgl.data
from utils.data_loader import *

import logging


"""fix the random seed"""
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""read args"""
global args, device
args = parse_args()
device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")



data_file='/HGA/invoke_edge'

train_g, train_pos_g, train_neg_g,test_pos_g,test_neg_g,all_nodes,number_mashup,g = load_data(args,data_file)


from dgl.data.utils import save_graphs

graph_labels = {"glabel": torch.tensor([0,1,2,3,4,5])}
save_graphs("./data/HGA/graph.bin", [train_g, train_pos_g, train_neg_g,test_pos_g,test_neg_g], graph_labels)




import json
def encode_sample(sample_dict):
    # 从字典中提取所有特征值并转换为 1D 张量
    feature_tensors = [torch.tensor(value, dtype=torch.float32).flatten() for value in sample_dict.values()]
    # 将所有特征张量拼接为一个向量
    combined_vector = torch.cat(feature_tensors, dim=0)
    # print(combined_vector.shape)
    return combined_vector
        
# 将内容编码处理为 tensor,以便批次处理
m_encoded_data = [encode_sample(sample) for sample in mashup_em]
# print(len(encoded_data))
mashup_content = torch.stack(m_encoded_data)


a_encoded_data = [encode_sample(sample) for sample in api_em]
# print(len(encoded_data))
api_content = torch.stack(a_encoded_data)
