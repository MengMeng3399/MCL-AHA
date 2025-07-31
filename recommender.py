import sys
sys.argv = ['run.py']

import itertools
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
from torch.nn.parameter import Parameter
import numpy as np
from time import time
from prettytable import PrettyTable
import logging
from utils.parser import parse_args
# from utils.preprocess import *
import dgl
import dgl.data
import json
from torch.nn.utils.rnn import pad_sequence
from model import *


import random
seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



global args
args = parse_args()
device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")


# 数据读入
from dgl.data.utils import load_graphs
glist,_ = load_graphs("./data/PWA/graph.bin") 
# bi_glist,_ = load_graphs("./data/PWA/bi_graph.bin") 
train_g,train_pos_g,train_neg_g,test_pos_g,test_neg_g = glist[:5]

mashup_em = torch.load("./data/PWA/mashup_em.pt")
api_em = torch.load("./data/PWA/api_em.pt")


#n-H 超边的数量
model = Recommender(n_params, args,mashup_em,api_em,n_H=64)# .to(device)
# pred = MLPPredictor(256)
pred = MLPPredictor(int(64))

optimizer = torch.optim.Adam(
    itertools.chain(model.parameters(), pred.parameters()), lr=args.lr
)


def compute_mad(embeddings, normalize=True):
    """
    embeddings: Tensor of shape [N, d]
    normalize: Whether to L2-normalize before computing distances
    """
    if normalize:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    n = embeddings.shape[0]
    diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # [N, N, d]
    dist_matrix = torch.norm(diff, dim=2)  # [N, N]

    mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
    mad = dist_matrix[mask].mean()
    return mad.item()


# ----------- training -------------------------------- #
all_logits = []
for e in range(100):
    # forward
    h,loss_contrast,anti_loss= model(train_g)
    
    # 使用lightGCN
    # h,loss_contrast = model(tra_adj)
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss,bcr_loss= compute_loss(pos_score, neg_score,loss_contrast,anti_loss,h,n_params['n_mashup'],args)

#     # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % 5 == 0:
        print("In epoch {}, loss: {}".format(e, loss))

# ----------- check results ------------------------ #
from sklearn.metrics import roc_auc_score

with torch.no_grad():
    # h_user = h[:n_params['n_mashup']]      # mashup嵌入
    # h_item = h[n_params['n_mashup']:]       # api嵌入

    # mad_user = compute_mad(h_user)
    # mad_item = compute_mad(h_item)

    # print(f"MAD (User): {mad_user:.4f}, MAD (Item): {mad_item:.4f}")
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h) 
    score=pred(all_test_graph,h)

    hit,NDCG ,F1,AUC= evaluation_metrics(all_test_graph,test_pos_g,score,n_params,pos_score,neg_score,top_k=30)
    print("train loss is {}, val hit@k is {}, val ndcg is {}, val f1 is {}, val auc is {}".format(bcr_loss, hit, NDCG, F1, AUC))



    #spaese

def robust_compute_hit_ndcg_by_item_bin(
    item_counts,
    test_pos_dict,
    user_item_dict,
    score_dict,
    bins,
    top_k=10,
    verbose=True
):
    # Step 1: Bin items by interaction count
    bin_items_dict = defaultdict(list)
    unbinned_items = set()

    for item, cnt in item_counts.items():
        matched = False
        for i, (low, high) in enumerate(bins):
            if low <= cnt < high:
                bin_items_dict[i].append(item)
                matched = True
                break
        if not matched:
            unbinned_items.add(item)

    if verbose:
        print(f"\n[Bin Construction]")
        for i, (low, high) in enumerate(bins):
            print(f"Bin {i}: range=({low},{high}), num_items={len(bin_items_dict[i])}")
        print(f"Items not assigned to any bin: {len(unbinned_items)}")

    # Step 2: Evaluate per bin
    result = {}
    for bin_id, items in bin_items_dict.items():
        hits, ndcgs = [], []
        gt_hit_count = 0
        for u, gt_item in test_pos_dict.items():
            if gt_item not in items:
                continue
            if u not in user_item_dict:
                if verbose:
                    print(f"[Warning] User {u} not in user_item_dict.")
                continue
            candidate_items = user_item_dict[u]
            if gt_item not in candidate_items:
                if verbose:
                    print(f"[Skip] GT item {gt_item} not in candidate list for user {u}")
                continue

            # Score and sort
            scored = [(i, score_dict.get((u, i), -1e10)) for i in candidate_items]
            ranked = sorted(scored, key=lambda x: x[1], reverse=True)
            top_items = [i for i, _ in ranked[:top_k]]

            # Eval
            if gt_item in top_items:
                hits.append(1)
                rank = top_items.index(gt_item)
                ndcgs.append(1 / np.log2(rank + 2))
            else:
                hits.append(0)
                ndcgs.append(0)
            gt_hit_count += 1

        result[bin_id] = {
            'num_items': len(items),
            'eval_users': gt_hit_count,
            f'hit@{top_k}': np.mean(hits) if hits else 0.0,
            f'ndcg@{top_k}': np.mean(ndcgs) if ndcgs else 0.0
        }

    return result