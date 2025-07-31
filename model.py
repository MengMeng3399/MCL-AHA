import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum
from torch.nn.parameter import Parameter
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
from sklearn.metrics import precision_score,recall_score,ndcg_score


def edge_dropout(graph, drop_prob):
    # 获取原始边数
    num_edges = graph.num_edges()
    # 生成保留边的 mask（1 表示保留，0 表示丢弃）
    mask = torch.rand(num_edges, device=graph.device) > drop_prob
    # 用 mask 选出要保留的边
    edge_ids = mask.nonzero(as_tuple=False).squeeze()
    # edge_ids = torch.nonzero(mask, as_tuple=False).squeeze()
    # 构造新的子图（只保留被选中的边）
    subgraph = dgl.edge_subgraph(graph, edge_ids, relabel_nodes=False)
    return subgraph


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    return roc_auc_score(labels, scores)


def top_k_list(g,score,n_mashup,n_api,top_k=10):
    matrix=np.zeros((n_mashup,n_api))
    src,dst=g.edges()

    for i in range(len(src)):
        matrix[src[i].item(),dst[i].item()-n_mashup]=score[i].item()
    
    top_k_index=np.argpartition(matrix, -top_k, axis=1)[:, -top_k:]
        
    return top_k_index

def NDCG_k(g,pos_g,score,data_config,top_k):
   
    n_mashup = data_config['n_mashup']
    n_api = data_config['n_api']

    src1,dst1=pos_g.edges()
    label_matrix=np.zeros((n_mashup,n_api))

    for i in range(len(src1)):
        label_matrix[src1[i].item(),dst1[i].item()-n_mashup]=1

    src2,dst2=g.edges()
    score_matrix=np.zeros((n_mashup,n_api))
    for i in range(len(src2)):
        if score[i].item()<0:
            score_matrix[src2[i].item(),dst2[i].item()-n_mashup]=0
        else:
            score_matrix[src2[i].item(),dst2[i].item()-n_mashup]=score[i].item()

    ndcg=ndcg_score(label_matrix,score_matrix,k=top_k)
    return ndcg


def NDCG_k2(g,pos_g,score,data_config,top_k):

    src1,dst1 = pos_g.edges()
    unique_u = torch.unique(src1)
    unique_u=unique_u.tolist()
    mapped_u = {value: index for index, value in enumerate(unique_u)}

    n_mashup = len(unique_u)
    all_mashup = data_config['n_mashup']
    n_api = data_config['n_api']

    label_matrix=np.zeros((n_mashup,n_api))

    for i in range(len(src1)):
        label_matrix[mapped_u[src1[i].item()],dst1[i].item()-all_mashup]=1

    src2,dst2=g.edges()

    score_matrix=np.zeros((n_mashup,n_api))
    for i in range(len(src2)):
        if score[i].item()<0:
            score_matrix[mapped_u[src2[i].item()],dst2[i].item()-all_mashup]=0
        else:
            score_matrix[mapped_u[src2[i].item()],dst2[i].item()-all_mashup]=score[i].item()

    ndcg=ndcg_score(label_matrix,score_matrix,k=top_k)
    return ndcg


def hit_k(g,pos_g,score,data_config,top_k):

    n_mashup = data_config['n_mashup']
    n_api = data_config['n_api']

    top_k_index = top_k_list(g,score,n_mashup,n_api,top_k)
    
    src,dst=pos_g.edges()
    matrix=np.zeros((n_mashup,n_api))
    for i in range(len(src)):
        matrix[src[i].item(),dst[i].item()-n_mashup]=1
    
    # 计算hit@k
    R_rate=0    
    n=0
    for i in range(top_k_index.shape[0]):
        hit_count=0
        for j in top_k_index[i]:
            if matrix[i][j]==1:
                hit_count+=1
        if np.sum(matrix[i])!=0:
            n+=1
            R_rate+=hit_count/np.sum(matrix[i])
    hit = R_rate/n
    print(n)
    return hit

def compute_f1(pos_score,neg_score):

    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()
    # >0.5 则预测正确
    scores=[1 if i>=0.0 else 0 for i in scores]
    f1=f1_score(labels, scores) 

    # precision=precision_score(labels, scores) 
    # recall=recall_score(labels, scores)
    # if (precision+recall)!=0:
    #     f1=2*precision*recall/(precision+recall)
    # else:
    #     f1=0
   
    return f1
    

def evaluation_metrics(g,pos_g,score,data_config,pos_score, neg_score,top_k):

    hit=hit_k(g,pos_g,score,data_config,top_k)
    NDCG=NDCG_k2(g,pos_g,score,data_config,top_k)

    F1=compute_f1(pos_score,neg_score)
    AUC=compute_auc(pos_score, neg_score)

    return hit , NDCG ,F1,AUC


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        为每个边产生一个标量分数.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src["h"], edges.dst["h"]], 1)
        
        # 这里为什么要使用squeeze(1)?----消除不必要的维度
        return {"score": self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(self.apply_edges)
            return g.edata["score"]


# 为lightGCN构建矩阵，为了加速卷积速度，提前计算邻接矩阵
def get_norm_adj_mat(g,n_mashup,n_api,n_nodes):
    # 创建一个 空的稀疏矩 ，并将其转换为lil_matrix 类型
    adjacency_matrix = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)
    adjacency_matrix = adjacency_matrix.tolil()
    # 根据g中的边计算邻接矩阵
    u,v=g.edges()
    R= sp.dok_matrix((n_mashup, n_api), dtype=np.float32)

    for row, col in zip(u, v):
        R[row, col-n_mashup] = 1
        
    '''
        [ 0  R]
        [R.T 0]
    '''
    adjacency_matrix[:n_mashup, n_mashup:] = R
    adjacency_matrix[n_mashup:, :n_mashup] = R.T
    adjacency_matrix = adjacency_matrix.todok()
 
    row_sum = np.array(adjacency_matrix.sum(axis=1))
     
    d_inv = np.power(row_sum, -0.5).flatten()

    d_inv[np.isinf(d_inv)] = 0.
    degree_matrix = sp.diags(d_inv)

    # D^(-1/2) A D^(-1/2)
    norm_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
    # 为了后续的计算，直接将其转换为tensor类型的稀疏矩阵
    # norm_adjacency= convert_sp_mat_to_sp_tensor(norm_adjacency)

    return norm_adjacency

# ----------------计算的是节点表示----------------------
from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        #GraphConv(输入shape，输出shape,聚合函数的类型，...)
        self.conv1 = SAGEConv(in_feats, h_feats, "mean")
        self.conv2 = SAGEConv(h_feats, h_feats, "mean")

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
class HyperGraphConv(nn.Module):
    def __init__(self, dim_in, dim_out, n_hyperedges, mode='dot', temperature=1.0, dropout_rate=0.2):
        super(HyperGraphConv, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_H = n_hyperedges
        self.mode = mode
        self.temperature = temperature
        self.dropout_rate = dropout_rate

        # 可训练超边表示
        self.hyper = nn.Parameter(torch.empty(n_hyperedges, dim_in))
        nn.init.xavier_uniform_(self.hyper)

        # 超边权重计算方式
        if mode == 'mlp':
            self.edge_mlp = nn.Sequential(
                nn.Linear(4 * dim_in, dim_in),
                nn.ReLU(),
                nn.Linear(dim_in, 1)
            )

        # 节点线性变换
        self.linear = nn.Linear(dim_in, dim_out)

    def compute_weights(self, node_em, hyper_emb):
        N_node = node_em.size(0)
        hyper_expand = hyper_emb.unsqueeze(0).expand(N_node, -1, -1)  # [N_node, N_H, dim]
        node_expand = node_em.unsqueeze(1)  # [N_node, 1, dim]

        if self.mode == 'dot':
            logits = torch.bmm(node_expand, hyper_expand.transpose(1, 2)).squeeze(1)  # [N_node, N_H]
        elif self.mode == 'mlp':
            node_expand = node_em.unsqueeze(1).expand(-1, self.n_H, -1)
            concat = torch.cat([
                hyper_expand,
                node_expand,
                hyper_expand * node_expand,
                torch.abs(hyper_expand - node_expand)
            ], dim=-1)
            logits = self.edge_mlp(concat).squeeze(-1)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return F.softmax(logits, dim=1)

    def forward(self, node_em, weights_override=None):
        """
        node_em: [N_node, dim_in]
        weights_override: [N_node, N_H] (optional) - externally masked weights
        """
        # Dropout on hyperedge embeddings
        hyper_dropout = F.dropout(self.hyper, p=self.dropout_rate, training=self.training)

        # 权重计算 or 使用外部传入
        if weights_override is None:
            weights = self.compute_weights(node_em, hyper_dropout)  # [N_node, N_H]
        else:
            weights = weights_override

        # 超图双向传播
        hyper_emb = torch.matmul(weights.T, node_em)     # [N_H, dim_in]
        updated_node = torch.matmul(weights, hyper_emb)  # [N_node, dim_in]

        # 映射输出维度
        updated_node = self.linear(updated_node)

        return F.relu(updated_node)
    
class Recommender(nn.Module):
    
    def __init__(self, data_config, args_config,mashup_em,api_em,n_H):
        super(Recommender, self).__init__()

        self.n_mashup = data_config['n_mashup']
        self.n_api = data_config['n_api']
        self.n_nodes = data_config['n_nodes'] 
        self.dim_m = data_config['dim_mem']
        self.dim_a = data_config['dim_aem']
        # args_config.dim = 64
        self.emb_size = 32
        # lightGCN的层数
        # self.lightgcn_layer = 2
        # 卷积层
        self.gnn_layer = 3
        self.hy_gnn_layer= 3
        self.keep_rate =0.80
        # mashup-mashup需求超图 和 api-api互补超图的的 侧信息初始化嵌入
        self.mashup_em = mashup_em
        self.api_em = api_em

        #超边的数量
        self.n_H = n_H

        # 初始化 id嵌入 
        self.all_embed = nn.Parameter(torch.empty(self.n_nodes, self.emb_size))
        nn.init.xavier_uniform_(self.all_embed)

        # 初始化，图卷积网络： 测试阶段为了速度快点 先使用GraphSAGE dgl封装好的卷积
        self.invoke_GraphSAGE = GraphSAGE(self.emb_size, self.emb_size)
        # 初始化 内容提取器 双层mlp：
        self.mmcontent = nn.Sequential(
            nn.Linear(self.dim_m, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            )
        
        self.aacontent = nn.Sequential(
            nn.Linear(self.dim_a, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            )
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            )
        
        self.fc2 = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            )
        # 初始化 超图卷积：
        self.mm_HyperGraphConv = HyperGraphConv(self.emb_size,self.emb_size,self.n_H)
        self.aa_HyperGraphConv = HyperGraphConv(self.emb_size,self.emb_size,self.n_H,mode='mlp')

    # 计算两个输入向量之间的余弦相似度 ---输入向量归一化处理
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def contrastive_m_loss(self,A_embedding,B_embedding):
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc1(A_embedding)
        B_embedding = self.fc1(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret 
    
    def contrastive_a_loss(self,A_embedding,B_embedding):
        tau = 0.6    # default = 0.8
        f = lambda x: torch.exp(x / tau)
        A_embedding = self.fc2(A_embedding)
        B_embedding = self.fc2(B_embedding)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret
    
    def compute_soft_anti_constraint(self,hyper_weights, sim_threshold=0.8):
        """
        api_init_emb: [N_api, dim]      初始化的 API 表示
        hyper_weights: [N_api, N_H]     当前 API 对超边的连接权重
        sim_threshold: float            对相似度施加 mask
        return: scalar loss
        """
        init_norm = F.normalize(self.api_em, dim=1)
        sim_matrix = torch.matmul(init_norm, init_norm.T)  # [N, N]

        # 2. 掩码（只对“看起来语义冗余”的API对加惩罚）
        mask = (sim_matrix > sim_threshold).float()
        sim_matrix = sim_matrix * mask  # 筛掉弱相关pair

        # 3. 超边连接相似度
        weight_norm = F.normalize(hyper_weights, dim=1)
        weight_sim = torch.matmul(weight_norm, weight_norm.T)  # [N, N]

        # 4. 加权惩罚项
        anti_matrix = sim_matrix * weight_sim

        # 5. 排除对角线
        anti_matrix = anti_matrix - torch.diag(torch.diag(anti_matrix))

        return anti_matrix.mean()
    
    def generate_masked_hypergraph(self, H, drop_prob=0.2):
        """超图mask扰动机制,用于对比学习增强"""
        mask = (torch.rand_like(H) > drop_prob).float()
        return H * mask
    
    def masked_hyper_contrastive(self, base_input, hyper_layer, drop_prob=0.2, temperature=0.5):
        # 1. 计算动态超边权重 W
        weights = hyper_layer.compute_weights(base_input, hyper_layer.hyper)

        # 2. 生成两个 masked 权重矩阵（mask 超边连接）
        mask1 = (torch.rand_like(weights) > drop_prob).float()
        mask2 = (torch.rand_like(weights) > drop_prob).float()
        weights1 = weights * mask1
        weights2 = weights * mask2

        # 3. 超图卷积输出（两个视图）
        z1 = hyper_layer(base_input, weights_override=weights1)
        z2 = hyper_layer(base_input, weights_override=weights2)

        # 4. 对比损失（InfoNCE）
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        positives = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
        negatives = torch.exp(sim_matrix).sum(dim=1)
        cl_loss = -torch.log(positives / negatives).mean()

        return cl_loss
        
    def forward(self, g):
        # 第一层初始嵌入
        # invoke_emb = self.invoke_GraphSAGE(g, self.all_embed)
        invoke_m_emb = self.all_embed[:self.n_mashup, :]
        invoke_a_emb = self.all_embed[self.n_mashup:, :]
        # 内容提取器输出（用于超图传播）
        mashup_content = self.mmcontent(self.mashup_em)
        api_content = self.aacontent(self.api_em)

        # GNN
        gnnU = [invoke_m_emb]
        gnnI = [invoke_a_emb]
        for _ in range(self.gnn_layer):
            if self.training:
                g_drop = edge_dropout(g, drop_prob=1 - self.keep_rate)
            else:
                g_drop = g

            node_lat = self.invoke_GraphSAGE(g_drop, torch.cat([gnnU[-1], gnnI[-1]], dim=0))
            um_lat = node_lat[:self.n_mashup, :]
            ia_lat = node_lat[self.n_mashup:, :]

            # 残差更新
            gnnU.append(um_lat + gnnU[-1])
            gnnI.append(ia_lat + gnnI[-1])

        final_gnn_um = torch.stack(gnnU, dim=0).sum(dim=0)
        final_gnn_ia = torch.stack(gnnI, dim=0).sum(dim=0)

        # hypergraph
        mashup_input = invoke_m_emb + mashup_content
        api_input = invoke_a_emb + api_content
        hyperU = [mashup_input]
        hyperI = [api_input]

        for _ in range(self.hy_gnn_layer):  
            m_lat = self.mm_HyperGraphConv(hyperU[-1])
            a_lat = self.aa_HyperGraphConv(hyperI[-1])
            hyperU.append(m_lat + hyperU[-1])  # 残差更新
            hyperI.append(a_lat + hyperI[-1])

        final_hyper_um = torch.stack(hyperU, dim=0).sum(dim=0)
        final_hyper_ia = torch.stack(hyperI, dim=0).sum(dim=0)
    
        # # 最终的嵌入
        final_um_lat = torch.cat([final_gnn_um, final_hyper_um], dim=-1)
        final_ia_lat = torch.cat([final_gnn_ia, final_hyper_ia], dim=-1)

        # # 超图mask对比学习 
        mask_cl_m = self.masked_hyper_contrastive(final_hyper_um, self.mm_HyperGraphConv)
        mask_cl_a = self.masked_hyper_contrastive(final_hyper_ia, self.aa_HyperGraphConv)

        # # 对比损失（使用图 vs 超图）
        m_loss = self.contrastive_m_loss(final_gnn_um, final_hyper_um)
        a_loss = self.contrastive_a_loss(final_gnn_ia, final_hyper_ia)

        # # # 互斥性损失（保持不变）
        api_hyper_weights = self.aa_HyperGraphConv.compute_weights(
            final_hyper_ia, self.aa_HyperGraphConv.hyper
        )

        # # # 传入 anti-constraint loss 函数
        anti_loss = self.compute_soft_anti_constraint(api_hyper_weights)

        # # 拼接图和超图信息作为最终嵌入输出
        node_emd = torch.cat([final_um_lat, final_ia_lat], dim=0)

        return node_emd,  (m_loss + a_loss)+0.5*(mask_cl_m + mask_cl_a) , anti_loss
    
        # return node_emd, m_loss + a_loss+mask_cl_m + mask_cl_a, anti_loss
        
        # return node_emd, m_loss + a_loss+mask_cl_m + mask_cl_a, anti_loss
        

def compute_loss(pos_score, neg_score,loss_contrast,anti_loss,h,n_mashup,data_args):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    )
    decay=data_args.l2
    # bpr
    criteria = nn.BCEWithLogitsLoss()
    bce_loss = criteria(scores, labels)

    mashup_emb=h[:n_mashup,:]
    api_emb=h[n_mashup:,:]
    # L2范数的平方和
    regularizer = (torch.norm(mashup_emb) ** 2
                    + torch.norm(api_emb) ** 2) / 2
    emb_loss = decay * regularizer / n_mashup

    # 加入正则项 鼓励 不同功能的
    return bce_loss+0.1*loss_contrast+emb_loss + 0.8*anti_loss , bce_loss+emb_loss

