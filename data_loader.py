import pandas as pd
import os
import scipy.sparse as sp
import dgl
import dgl.data
import numpy as np
import torch
import json
import math
import random
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, LabelBinarizer
from multiprocessing import Queue, Process
import ast
import torch.nn.functional as F

def bulid_invoke_graph(invoke_data):
    # 传进来的数据是dataframe
    src=invoke_data.iloc[:,0].to_numpy()
    rel=invoke_data.iloc[:,1].to_numpy()
    dst=invoke_data.iloc[:,2].to_numpy()

    all_nodes=np.max(src)+np.max(dst)+1+1
    number_mashup=np.max(src)+1
    # mashup节点:[:number_mashup]   api节点:[number_mashup+1:]
    dst=dst+number_mashup
    g = dgl.graph((src, dst))
    return g ,all_nodes,number_mashup
    

def Subgraph_generation(g):
    #--------------------------------正样本----------------------
    u, v = g.edges()
    #0,1,...,num_edges-1
    eids = np.arange(g.num_edges())
    #对序列进行随机排序--形成一个随机列表  
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * 0.1)
    train_size = g.num_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]
    
    #--------------------------------负样本-----------------------------
    # 查找所有的负样本
    # 压缩矩阵sp.coo_matrix((data,(row,col)),...) --->(row,col)位置的值为data
    # 将存在边的位置设为1，shape = num_nodes * num_nodes   ---此时对角线处的值为0
    ratio=10
    adj = sp.coo_matrix(( np.ones(len(u)), (u.numpy(), v.numpy())),shape=(g.num_nodes(),g.num_nodes()) )
    # todense()将稀疏矩阵转换为稠密的矩阵(np.matrix)
    adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
    # np.where(adj_neg != 0) 返回的是值不为0的row和col
    neg_u, neg_v = np.where(adj_neg != 0)

    #从没有边的数据中选取num_edges
    neg_eids = np.random.choice(len(neg_u), g.num_edges()*ratio)
    neg_test_size=int(len(neg_eids) * 0.1)

    test_neg_u, test_neg_v = (
        neg_u[neg_eids[:neg_test_size]],
        neg_v[neg_eids[:neg_test_size]],
    )
    train_neg_u, train_neg_v = (
        neg_u[neg_eids[neg_test_size:]],
        neg_v[neg_eids[neg_test_size:]],
    )

    # 生成子图:-------单向图
    # 1.train_g : 从g中移除test中的边
    #remove_edges(图，要移除的边的ID)  ----不会改变节点的ID
    train_g = dgl.remove_edges(g, eids[:test_size])

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

    return train_g, train_pos_g,train_neg_g,test_pos_g,test_neg_g

def generate_unique_random_not_in_list(A,start,end,n):
    temp_list=[]
    count = 0
    flag = True
    while flag:
        random_num = random.randint(start, end-1)  # 生成一个 (start, end) 范围内的随机数
        if count < n:
            if random_num not in A and random_num not in temp_list:
                temp_list.append(random_num)
                count+=1
        else:
            return temp_list
        
def subgraph_invoke(g, num_mashup, all_nodes, test_ratio=0.2, neg_sample_size=5):
    u, v = g.edges()
    u = u.tolist()
    v = v.tolist()

    mashup_nodes = list(range(num_mashup))
    api_nodes = list(range(num_mashup, all_nodes))

    edge_list = list(zip(u, v))
    random.shuffle(edge_list)

    # 将边按 mashup 分组
    mashup_to_edges = {m: [] for m in mashup_nodes}
    for edge in edge_list:
        m, a = edge
        if m in mashup_to_edges:
            mashup_to_edges[m].append(edge)

    # 为每个 mashup 选择一条边作为 test 保底（确保每个 mashup 都在 test 中）
    test_edges = []
    remaining_edges = []

    for m in mashup_nodes:
        edges = mashup_to_edges[m]
        if len(edges) == 0:
            continue  # 极端情况处理（若某 mashup 无边）
        random.shuffle(edges)
        test_edges.append(edges[0])              # 保留一条做 test
        remaining_edges.extend(edges[1:])        # 剩下的加入待划分集合

    # 计算需要的测试集大小
    total_test_size = int(len(edge_list) * test_ratio)
    extra_needed = max(0, total_test_size - len(test_edges))  # 已有基础上扩展

    # 从剩余边中再随机选择若干 test
    random.shuffle(remaining_edges)
    test_edges += remaining_edges[:extra_needed]
    train_edges = remaining_edges[extra_needed:]

    # 划分正样本边
    train_pos_u, train_pos_v = zip(*train_edges) if train_edges else ([], [])
    test_pos_u, test_pos_v = zip(*test_edges)

    train_pos_g = dgl.graph((torch.tensor(train_pos_u), torch.tensor(train_pos_v)), num_nodes=all_nodes)
    test_pos_g = dgl.graph((torch.tensor(test_pos_u), torch.tensor(test_pos_v)), num_nodes=all_nodes)

    train_g = train_pos_g

    # 负采样函数
    def negative_sampling(pos_edges, existing_edges_set):
        neg_u, neg_v = [], []
        for m_id, _ in pos_edges:
            for _ in range(neg_sample_size):
                while True:
                    sampled_api = random.choice(api_nodes)
                    if (m_id, sampled_api) not in existing_edges_set:
                        neg_u.append(m_id)
                        neg_v.append(sampled_api)
                        break
        return torch.tensor(neg_u), torch.tensor(neg_v)

    train_edge_set = set(train_edges)
    test_edge_set = set(test_edges)
    all_edge_set = set(edge_list)

    train_neg_u, train_neg_v = negative_sampling(train_edges, all_edge_set)
    test_neg_u, test_neg_v = negative_sampling(test_edges, all_edge_set)

    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=all_nodes)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=all_nodes)

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g


def Subgraph_cold_v2(g, num_mashup, all_nodes, cold_ratio=0.2, test_train_ratio=0.1, neg_ratio=3):
    # neg_ratio 正负样本比列
    u, v = g.edges()

    mashup_nodes = list(range(num_mashup))
    api_nodes = list(range(num_mashup, all_nodes))

    # 采样冷启动mashup和冷启动API
    cold_mashups = set(random.sample(mashup_nodes, int(cold_ratio * len(mashup_nodes))))
    cold_apis = set(random.sample(api_nodes, int(cold_ratio * len(api_nodes))))

    warm_mashups = set(mashup_nodes) - cold_mashups
    warm_apis = set(api_nodes) - cold_apis

    # 分类边
    mashup_cold_edges = []
    api_cold_edges = []
    both_cold_edges = []
    warm_edges = []

    for idx in range(len(u)):
        src = u[idx].item()
        dst = v[idx].item()
        
        if src in cold_mashups and dst in cold_apis:
            both_cold_edges.append(idx)
        elif src in cold_mashups and dst in warm_apis:
            mashup_cold_edges.append(idx)
        elif src in warm_mashups and dst in cold_apis:
            api_cold_edges.append(idx)
        elif src in warm_mashups and dst in warm_apis:
            warm_edges.append(idx)

    # warm_edges进一步划分：90%训练，10%热启动测试
    warm_edges = np.array(warm_edges)
    np.random.shuffle(warm_edges)

    warm_test_size = int(test_train_ratio * len(warm_edges))
    warm_test_edges = warm_edges[:warm_test_size]
    warm_train_edges = warm_edges[warm_test_size:]

    # 正样本集合
    train_pos_u, train_pos_v = u[warm_train_edges], v[warm_train_edges]

    mashup_cold_pos_u, mashup_cold_pos_v = u[mashup_cold_edges], v[mashup_cold_edges]
    api_cold_pos_u, api_cold_pos_v = u[api_cold_edges], v[api_cold_edges]
    both_cold_pos_u, both_cold_pos_v = u[both_cold_edges], v[both_cold_edges]
    warm_test_pos_u, warm_test_pos_v = u[warm_test_edges], v[warm_test_edges]

    #-------------------------负样本采样-------------------------
    train_neg_u, train_neg_v = [], []
    mashup_cold_neg_u, mashup_cold_neg_v = [], []
    api_cold_neg_u, api_cold_neg_v = [], []
    both_cold_neg_u, both_cold_neg_v = [], []
    warm_test_neg_u, warm_test_neg_v = [], []

    # 创建邻接表方便负采样排除已有连接
    adj_dict = {}
    for i in range(len(u)):
        src, dst = u[i].item(), v[i].item()
        if src not in adj_dict:
            adj_dict[src] = set()
        adj_dict[src].add(dst)

    def generate_negatives(src_nodes, candidate_nodes, sample_size):
        neg_v = []
        for src in src_nodes:
            avoid = adj_dict.get(src, set())
            samples = generate_unique_random_not_in_list(list(avoid), min(candidate_nodes), max(candidate_nodes)+1, sample_size)
            neg_v += samples
        return neg_v

    # Train负样本（warm mashup->warm api）
    train_neg_u = []
    train_neg_v = []
    for src in train_pos_u.tolist():
        train_neg_u += [src] * neg_ratio
        train_neg_v += generate_unique_random_not_in_list(list(adj_dict.get(src, set())), num_mashup, all_nodes, neg_ratio)

    # Warm测试负样本
    for src in warm_test_pos_u.tolist():
        warm_test_neg_u += [src] * neg_ratio
        warm_test_neg_v += generate_unique_random_not_in_list(list(adj_dict.get(src, set())), num_mashup, all_nodes, neg_ratio)

    # mashup冷测试负样本（冷mashup -> 热api）
    for src in mashup_cold_pos_u.tolist():
        mashup_cold_neg_u += [src] * neg_ratio
        mashup_cold_neg_v += generate_unique_random_not_in_list(list(adj_dict.get(src, set())), num_mashup, all_nodes, neg_ratio)

    # api冷测试负样本（热mashup -> 冷api）
    for src in api_cold_pos_u.tolist():
        api_cold_neg_u += [src] * neg_ratio
        api_cold_neg_v += generate_unique_random_not_in_list(list(adj_dict.get(src, set())), num_mashup, all_nodes, neg_ratio)

    # 双冷负样本（冷mashup -> 冷api）
    for src in both_cold_pos_u.tolist():
        both_cold_neg_u += [src] * neg_ratio
        both_cold_neg_v += generate_unique_random_not_in_list(list(adj_dict.get(src, set())), num_mashup, all_nodes, neg_ratio)

    #--------------------------------------子图生成----------------------------------
    # 训练集
    train_g = dgl.remove_edges(g, warm_test_edges.tolist() + mashup_cold_edges + api_cold_edges + both_cold_edges)
    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    train_neg_g = dgl.graph((torch.tensor(train_neg_u), torch.tensor(train_neg_v)), num_nodes=g.num_nodes())

    # 测试集
    warm_test_pos_g = dgl.graph((warm_test_pos_u, warm_test_pos_v), num_nodes=g.num_nodes())
    warm_test_neg_g = dgl.graph((torch.tensor(warm_test_neg_u), torch.tensor(warm_test_neg_v)), num_nodes=g.num_nodes())

    mashup_cold_pos_g = dgl.graph((mashup_cold_pos_u, mashup_cold_pos_v), num_nodes=g.num_nodes())
    mashup_cold_neg_g = dgl.graph((torch.tensor(mashup_cold_neg_u), torch.tensor(mashup_cold_neg_v)), num_nodes=g.num_nodes())

    api_cold_pos_g = dgl.graph((api_cold_pos_u, api_cold_pos_v), num_nodes=g.num_nodes())
    api_cold_neg_g = dgl.graph((torch.tensor(api_cold_neg_u), torch.tensor(api_cold_neg_v)), num_nodes=g.num_nodes())

    both_cold_pos_g = dgl.graph((both_cold_pos_u, both_cold_pos_v), num_nodes=g.num_nodes())
    both_cold_neg_g = dgl.graph((torch.tensor(both_cold_neg_u), torch.tensor(both_cold_neg_v)), num_nodes=g.num_nodes())

    return train_g, train_pos_g, train_neg_g, warm_test_pos_g, warm_test_neg_g, mashup_cold_pos_g, mashup_cold_neg_g, api_cold_pos_g, api_cold_neg_g, both_cold_pos_g, both_cold_neg_g



def Subgraph_generation_mashup(g):
    #--------------------------------正样本----------------------
    u, v = g.edges()
    test_train_ratio=0.1
    temp_dict={}
    for i in range(len(u)):
        if u[i].item() not in temp_dict:
            temp_dict[u[i].item()]=[i]
        else:
            temp_dict[u[i].item()].append(i)

    index=[]
    for key,value in temp_dict.items():
        #向上取整
        test_len = math.ceil(len(value) * test_train_ratio)
        if test_len > 0:
            test_eids = np.random.choice(value, test_len, replace=False)
            index.append(test_eids.tolist())

    edges_test = [item for sublist in index for item in sublist]
    edges_all = list(range(g.num_edges()))

    edges_train = list(set(edges_all) - set(edges_test))

    test_pos_u, test_pos_v = u[edges_test], v[edges_test]
    train_pos_u, train_pos_v = u[edges_train], v[edges_train]
        
    
    #--------------------------------负样本-----------------------------
    # 查找所有的负样本
    # 压缩矩阵sp.coo_matrix((data,(row,col)),...) --->(row,col)位置的值为data
    # 将存在边的位置设为1，shape = num_nodes * num_nodes   ---此时对角线处的值为0
    ratio=3
    adj = sp.coo_matrix(( np.ones(len(u)), (u.numpy(), v.numpy())),shape=(g.num_nodes(),g.num_nodes()) )
    # todense()将稀疏矩阵转换为稠密的矩阵(np.matrix)
    adj_neg = 1 - adj.todense() - np.eye(g.num_nodes())
    # np.where(adj_neg != 0) 返回的是值不为0的row和col
    neg_u, neg_v = np.where(adj_neg != 0)

    #从没有边的数据中选取num_edges
    neg_eids = np.random.choice(len(neg_u), g.num_edges()*ratio)
    neg_test_size=int(len(neg_eids) * 0.1)

    test_neg_u, test_neg_v = (
        neg_u[neg_eids[:neg_test_size]],
        neg_v[neg_eids[:neg_test_size]],
    )
    train_neg_u, train_neg_v = (
        neg_u[neg_eids[neg_test_size:]],
        neg_v[neg_eids[neg_test_size:]],
    )

    # 生成子图:-------单向图
    # 1.train_g : 从g中移除test中的边
    #remove_edges(图，要移除的边的ID)  ----不会改变节点的ID
    train_g = dgl.remove_edges(g, edges_test)

    train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.num_nodes())
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=g.num_nodes())

    test_pos_g = dgl.graph((test_pos_u, test_pos_v), num_nodes=g.num_nodes())
    test_neg_g = dgl.graph((test_neg_u, test_neg_v), num_nodes=g.num_nodes())

    return train_g, train_pos_g,train_neg_g,test_pos_g,test_neg_g

def data_preprocess(dataPath):
    '''
     parameter:

     nodes_data:  dataframe mashup/api 的特征

     return:
     all_node_feat:  字典的列表： 存储每个节点的每个特征
     
     '''
    
    featType_enabled = ['oneHot', 'multiHot']
    featDict = {
    'mashup' : {
        'oneHot' : ['ID', 'MashupName'],
        'multiHot' : ['MashupCategory'],
        'textual':['MashupDescription']
    },
    'api' : {
        'oneHot' : ['ID', 'ApiName'],
        'multiHot' : ['ApiTags'],
        'textual':['ApiDescription']
    }
    }

    mashup_datapath=dataPath +'/'+ 'mashup_data'
    if os.path.exists(mashup_datapath + '.csv'):
        mashup_data = pd.read_csv(mashup_datapath + '.csv')
    else:
        print("warn: mashup_data not found")

    api_datapath=dataPath +'/'+ 'api_data'
    if os.path.exists(api_datapath + '.csv'):
        api_data = pd.read_csv(api_datapath + '.csv')
    else:
        print("warn: API_data not found")
    
    mashup_bert_path = dataPath +'/'+ 'bert_mashup_des.json'
    api_bert_path = dataPath +'/'+ 'bert_api_des.json'

    with open(mashup_bert_path, 'r') as fd:
        mashupRawData = fd.read()
        mashup_bert_data = json.loads(mashupRawData)
    with open(api_bert_path, 'r') as fd:
        apiRawData = fd.read()
        api_bert_data = json.loads(apiRawData)
    #  -----------------------------------------------------Mashup进行编码
    sumdata = {} 
    for featType in featType_enabled:
        for feat in featDict['mashup'][featType]:
            value = mashup_data[feat].to_list()        
            sumdata[feat]=value
        
    encoderDict = {} # 编码器字典
    for feat in featDict['mashup']['oneHot']:
        encoderDict[feat] = LabelBinarizer()
        encoderDict[feat].fit(sumdata[feat])
    for feat in featDict['mashup']['multiHot']:
        encoderDict[feat] = MultiLabelBinarizer()
        encoderDict[feat].fit(sumdata[feat])


    # 进行编码：
    # 语法：transform这里返回的是二维nddary 
    mashup_node_feat=[0]*mashup_data.shape[0]
    for index, the_node in mashup_data.iterrows():
        node_feat={}
        # 首先将bert处理之后的文本特征保存
        node_feat['MashupDescription']=np.array(mashup_bert_data[the_node['MashupName']])
        for featType in featType_enabled:
            for feat in featDict['mashup'][featType]:  
                encode_nd = encoderDict[feat].transform([the_node[feat]])
                node_feat[feat]=encode_nd       
        mashup_node_feat[index]=node_feat


    #  -----------------------------------------------------api进行编码
    sumdata = {} 
    for featType in featType_enabled:
        for feat in featDict['api'][featType]:
            value = api_data[feat].to_list()        
            sumdata[feat]=value
        
    encoderDict = {} # 编码器字典
    for feat in featDict['api']['oneHot']:
        encoderDict[feat] = LabelBinarizer()
        encoderDict[feat].fit(sumdata[feat])
    for feat in featDict['api']['multiHot']:
        encoderDict[feat] = MultiLabelBinarizer()
        encoderDict[feat].fit(sumdata[feat])


    # 进行编码：
    # 语法：transform这里返回的是二维nddary 
    api_node_feat=[0]*api_data.shape[0]
    for index, the_node in api_data.iterrows():
        node_feat={}
        node_feat['ApiDescription']=np.array(api_bert_data[the_node['ApiName']])
        for featType in featType_enabled:
            for feat in featDict['api'][featType]:  
                encode_nd = encoderDict[feat].transform([the_node[feat]])
                node_feat[feat]=encode_nd
                
        api_node_feat[index]=node_feat

    return mashup_node_feat, api_node_feat


def load_data(model_args,data_file):

    args = model_args
    directory = args.data_path +'/'
    print('reading train and test mashup-api set ...')
    invoke_flies= args.data_path + data_file

    if os.path.exists(invoke_flies + '.csv'):
        invoke_data = pd.read_csv(invoke_flies + '.csv')
    else:
        print("warn: data invoke_edge not found")
    
    # 生成样本图
    g,all_nodes,number_mashup =bulid_invoke_graph(invoke_data)

    # 划分数据,构建四个子图:-  train_g ,train_pos_g, train_neg_g, test_pos_g, test_neg_g,
    # train_g, train_pos_g,train_neg_g,test_pos_g,test_neg_g=Subgraph_generation(g)
    # train_g, train_pos_g,train_neg_g,test_pos_g,test_neg_g=Subgraph_generation_mashup(g)

    # train_g, train_pos_g,train_neg_g,warm_test_pos_g,warm_test_neg_g,cold_test_pos_g,cold_test_neg_g,all_test_pos_g,all_test_neg_g = Subgraph_cold (g,number_mashup,all_nodes,cold_ratio)

    # train_g, train_pos_g, train_neg_g, warm_test_pos_g, warm_test_neg_g, mashup_cold_pos_g, mashup_cold_neg_g, api_cold_pos_g, api_cold_neg_g, both_cold_pos_g, both_cold_neg_g=Subgraph_cold_v2(g,number_mashup,all_nodes)
    train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g=subgraph_invoke(g,number_mashup,all_nodes)
    

    return train_g, train_pos_g, train_neg_g, test_pos_g, test_neg_g,all_nodes,number_mashup,g

#----------构建标签共现互补图--------------

def remap_items(ID,data_file,type='mashup',feat="MashupCategory"):
    # 去重,但是不改变顺序
    ID=list(dict.fromkeys(ID))
    tag_list=[]
    data= pd.read_csv(data_file)

    if type == "mashup":
        feat = "MashupCategory"
        
    elif type == 'api':
        feat = 'ApiTags'
    else:
        raise ValueError("Error: Type must be 'mashup' or 'api'")
    
    # 由于csv 将list转换为了字符串，因此需要将其改为list
    data[feat]=data[feat].apply(eval)
    temp_list=data.loc[ID,feat].tolist()

    for sublist in temp_list:
        for item in sublist:
            if item not in tag_list:
                tag_list.append(item)

    result_dict={value: index for index, value in enumerate(tag_list)}
    # train
    remap_dict={}
    for sub_data in ID:
        key=sub_data
        if key != data.loc[sub_data,'ID']:
            raise ValueError("Error: value error")
        value_list= []
        for sub_tag in data.loc[sub_data,feat]:
            value_list.append(result_dict[sub_tag])

        if key not in remap_dict:
            remap_dict[key]=value_list

    # all
    all_remap_dict={}
    for index, row in data.iterrows():
        key=row['ID']
        value_list= []
        for sub_tag in row[feat]:
            # train 中没有的标签,不做处理
            if sub_tag in result_dict:
                value_list.append(result_dict[sub_tag])

        if key not in all_remap_dict:
            all_remap_dict[key]=value_list

    return result_dict,remap_dict,all_remap_dict

# def tag_api_co_matrix(g,tag,n,m,x,offset):
#     '''
#     参数：
#     g: 图
#     tag:  字典  api_id:[tags_id]  ---train
#     n: 矩阵尺寸  tag矩阵
#     m: 矩阵尺寸  api矩阵
#     x: 字典  api_id: [tags_id]   ---all
#     '''
#     matrix=np.zeros((n,n))
#     temp_dict={}
#     u,v=g.edges()
#     # 构建 api 之间的共现list
#     for i in range(len(u)):
#         if u[i].item() not in temp_dict:
#             temp_dict[u[i].item()]=[v[i].item()-offset]
#         else:
#             temp_dict[u[i].item()].append(v[i].item()-offset)

#     for key,value in temp_dict.items():
#         for i in range(len(value)):
#             for j in range(i+1,len(value)):
#                 for k in 


def tag_co_matrix(g,tag_x,tag_y,rows,cols,n,all_nodes,x,y):

    u,v=g.edges()
    matrix=np.zeros((rows,cols))
    for i in range(len(u)):
        row=u[i].item()
        col=v[i].item()-n
        for sub_m in tag_x[row]:
            for sub_a in tag_y[col]:
                matrix[sub_m,sub_a]+=1
    
    # inverse 
    co_matrix=np.zeros((n,all_nodes-n))
    for i in range(n):
        for j in range(all_nodes-n):
            value=0
            for sub_m in x[i]:
                for sub_a in y[j]:
                    if value<matrix[sub_m,sub_a]:
                        value=matrix[sub_m,sub_a]
            co_matrix[i,j]=value
            
    return co_matrix,matrix





# 希望输入是0的部分，归一化之后依然是0.
def custom_softmax(x):
    y = np.zeros_like(x, dtype=float)
    for i in range(x.shape[0]):
        # 获取非零值的索引
        non_zero_indices = np.where(x[i] != 0)[0]
        # # 对非零值进行适当的缩放，减去最大值
        x_row = x[i, non_zero_indices] - np.max(x[i, non_zero_indices])
        # # 计算分母，避免指数溢出
        sum_exp = np.sum(np.exp(x_row))
        # # 计算 softmax
        for j, idx in enumerate(non_zero_indices):
            y[i][idx] = np.exp(x_row[j]) / sum_exp
    return y


def softmax(x):
    y = np.zeros_like(x, dtype=float)
    for i in range(x.shape[0]):
        # 进行适当的缩放，减去最大值
        x[i] -= np.max(x[i])
        sum_exp = np.sum(np.exp(x[i]))
        y[i]=np.exp(x[i])/sum_exp
        
        # for j in range(x.shape[1]):
        #     y[i][j] = np.exp(x[i][j]) / sum_exp
    return y


def custom_nor(x):
    y = np.zeros_like(x, dtype=float)
    for i in range(x.shape[0]):
        sum_i = np.sum((x[i]))
        if sum_i!=0:
            y[i]=x[i]/sum_i
    return y



# 将matrix转换为同构图，节点（0:rows）(rows:rows+cols)
def matrix_to_graph(matrix,flag_weight=False):
    nonzero_indices = np.nonzero(matrix)
    src=torch.tensor(nonzero_indices[0])
    dst=torch.tensor(nonzero_indices[1]+matrix.shape[0])
    g=dgl.graph((src,dst))

    if flag_weight:
        weight=torch.tensor([matrix[src[i].item(), dst[i].item()-matrix.shape[0]] for i in range(len(src))])
        g.edata['w']=weight
    return g

# 将matrix转换为同构图，节点（0:rows）(rows:rows+cols)
def dict_to_graph(offset,the_dict):
    src=[]
    dst=[]
    for key,value in the_dict.items():
        for j in value:
            src.append(key)
            dst.append(j+offset)

    src=torch.tensor(src)   
    dst=torch.tensor(dst)
    g=dgl.graph((src,dst))

    return g


def tag_subgraph(model_args,g,number_mashup,all_nodes,data_source):

    args = model_args
    number_api=all_nodes - number_mashup
    print('reading to generate complementary subgraphs for tag')
    mashup_files= args.data_path + data_source + '/mashup_data' + '.csv'
    api_files= args.data_path + data_source + '/api_data'+ '.csv'

    # 得到mashup和api的ID 
    mashup_ID,api_id=g.edges()
    api_ID=api_id-number_mashup
    # 
    m_tag_dict,m_remap,m_all_remap = remap_items(mashup_ID.tolist(),mashup_files)
    a_tag_dict,a_remap,a_all_remap = remap_items(api_ID.tolist(),api_files,type='api')
    rows=len(m_tag_dict)
    cols=len(a_tag_dict)
    print("num_mashup_tag",rows)
    print("num_api_tag",cols)

    # 由标签互补得到 mashup和api的 共现矩阵
    matrix,tag_matrix=\
        tag_co_matrix(g,m_remap,a_remap,rows,cols,number_mashup,all_nodes,m_all_remap,a_all_remap)
    
    # 为了对比模型，构建两个子图：
    

    # 进行归一化：
    # matrix=custom_softmax(matrix)
    # matrix=softmax(matrix)
    # matrix=custom_nor(matrix)

    # 
    tag_graph=matrix_to_graph(matrix,flag_weight='True')
    
    # k=10
    # top_k_indices = np.argpartition(matrix, -k, axis=1)[:, -k:]
    # tag_graph=matrix_index_to_graph(top_k_indices,number_mashup,k)
    # 去处替代关系的api-api子图 
    
    # matrix_api= tag_api_co_matrix(g,a_remap,cols,number_api,number_api)
    # tag_api_co_matrix(g,a_remap,cols,number_api,number_api,number_mashup)

    a_tag_graph = dict_to_graph(number_api,a_remap)
    m_tag_graph = dict_to_graph(number_mashup,m_remap)
    return tag_graph,a_tag_graph,m_tag_graph,tag_matrix,m_tag_dict,a_tag_dict,m_all_remap,a_all_remap
    # mashup的tags


    #-----------------构建语义互补图-----------------

def cosine_similarity(a, b):
    """
    计算两个向量的余弦相似度
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    
    return similarity


def remap_items_des(mashup_dict,api_dict,vec_mashup_des,vec_api_des,rows,cols):
    matrix=np.zeros((rows,cols))

    for i in range(rows):
        m_vec=np.array(vec_mashup_des[mashup_dict[i]])
        
        for j in range(cols):
            a_vec=np.array(vec_api_des[api_dict[j]])
            value=cosine_similarity(m_vec,a_vec)
            matrix[i,j]=value

    return matrix   


def bert_encode(mashup_dict,api_dict,vec_mashup_des,vec_api_des,num_m,num_a):
    # bert编码的长度
    emb_size=768
    m_bert_emd=torch.zeros(num_m, emb_size)
    a_bert_emd=torch.zeros(num_a, emb_size)

    for key,value in mashup_dict.items():
        m_emd = torch.tensor(vec_mashup_des[value])
        m_bert_emd[key]=m_emd

    for key,value in api_dict.items():
        a_emd = torch.tensor(vec_api_des[value])
        a_bert_emd[key]=a_emd
    
    bert_emd=torch.cat([m_bert_emd,a_bert_emd],dim=0)
        
    return bert_emd 


def matrix_index_to_graph(matrix,number_mashup,k,flag_weight=False):

    src = torch.arange(number_mashup).repeat(k)

    a=torch.tensor(matrix+number_mashup)

    dst=torch.cat([a[:, i] for i in range(a.shape[1])])

    g=dgl.graph((src,dst))

    return g


def semantic_graph(model_args,number_mashup,all_nodes,data_source,k=10):
    args = model_args
    print('reading to generate complementary subgraphs for semantic')
    mashup_files= args.data_path + data_source +'/mashup_data' + '.csv'
    api_files= args.data_path + data_source+ '/api_data'+ '.csv'

    mashup_data= pd.read_csv(mashup_files)
    api_data=pd.read_csv(api_files)

    mashup_dict=dict(zip(mashup_data['ID'], mashup_data['MashupName']))
    api_dict=dict(zip(api_data['ID'], api_data['ApiName']))

    mashup_des_files= args.data_path +data_source+'/bert_mashup_des' + '.json'
    api_des_files= args.data_path + data_source+'/bert_api_des'+ '.json'

    with open(mashup_des_files, 'r') as fd:
        mashupRawData = fd.read()
        vec_mashup_des = json.loads(mashupRawData)

    with open(api_des_files, 'r') as fd:
        apiRawData = fd.read()
        vec_api_des  = json.loads(apiRawData)

    rows=number_mashup
    cols=all_nodes-number_mashup

    matrix=remap_items_des(mashup_dict,api_dict,vec_mashup_des,vec_api_des,rows,cols)

    bert_emd= bert_encode(mashup_dict,api_dict,vec_mashup_des,vec_api_des,rows,cols)
    
    # 相似度 top_k
    top_k_indices = np.argpartition(matrix, -k, axis=1)[:, -k:]

    senmantic_g=matrix_index_to_graph(top_k_indices,number_mashup,k)
    
    return senmantic_g,bert_emd
    # return bert_emd

def Hyperedgeds(g,dataPath,num_mashup,num_api):
    src,dst = g.edges()
    # 得到api的id
    dst = dst - num_mashup

    mashup_datapath=dataPath +'/'+ 'mashup_data'
    if os.path.exists(mashup_datapath + '.csv'):
        mashup_data = pd.read_csv(mashup_datapath + '.csv')
    else:
        print("warn: mashup_data not found")

    api_datapath=dataPath +'/'+ 'api_data'
    if os.path.exists(api_datapath + '.csv'):
        api_data = pd.read_csv(api_datapath + '.csv')
    else:
        print("warn: API_data not found")
    
    # 1.2.超边co-service（事实上分装的是mashup）  与 co-mashup超图（封装的是api）
    H_cs, H_cm = build_co_service_and_co_mashup_hypergraph(g, num_mashup, num_api)

    # 3.4.Co-category
    H_m_cc = build_co_tag_hypergraph_dense(mashup_data,"MashupCategory")
    H_a_cc = build_co_tag_hypergraph_dense(api_data,"ApiTags")
    # 5. co-provider
    H_a_cp = build_co_provider_hypergraph_dense(api_data,"ApiTags")
    # 6 7 co-des
    m_bert_dict_list = dataPath +'/'+'bert_mashup_des.json'               # 原始 
    H_m_desc = build_co_description_hypergraph_by_id(m_bert_dict_list, mashup_data, top_k=3,the_type="mashup")

    a_bert_dict_list = dataPath +'/'+'bert_api_des.json'               # 原始 
    H_a_desc = build_co_description_hypergraph_by_id(a_bert_dict_list, api_data, top_k=3,the_type="api")

    H_m = torch.cat([H_cs, H_m_cc, H_m_desc], dim=1)

    H_s = torch.cat([H_cm, H_a_cc, H_a_cp,H_a_desc], dim=1)

    return H_m, H_s


def build_co_service_and_co_mashup_hypergraph(g, num_mashup, num_api):
    # 边：mashup → api
    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy() - num_mashup  # 将 API 节点 ID 转为 [0, num_api)

    # 构建调用矩阵 M: (num_mashup × num_api)
    M = torch.zeros((num_mashup, num_api), dtype=torch.float32)
    M[src, dst] = 1.0

    # ---- Co-Service 超图 封装的是mashup----
    # 对于每个 API j，被哪些 mashup 调用，就构成一个超边
    rows_cs, cols_cs = [], []
    edge_id = 0
    for j in range(num_api):
        mashup_ids = (M[:, j] == 1).nonzero(as_tuple=True)[0]
        if len(mashup_ids) >= 2:  # 至少2个构成超边
            for i in mashup_ids:
                rows_cs.append(i.item())
                cols_cs.append(edge_id)
            edge_id += 1

    H_cs = torch.sparse_coo_tensor(
        indices=[rows_cs, cols_cs],
        values=torch.ones(len(rows_cs)),
        size=(num_mashup, edge_id)
    ).to_dense()

    # ---- Co-Mashup 超图 ----
    rows_cm, cols_cm = [], []
    edge_id = 0
    for i in range(num_mashup):
        api_ids = (M[i, :] == 1).nonzero(as_tuple=True)[0]
        if len(api_ids) >= 2:
            for j in api_ids:
                rows_cm.append(j.item())
                cols_cm.append(edge_id)
            edge_id += 1

    H_cm = torch.sparse_coo_tensor(
        indices=[rows_cm, cols_cm],
        values=torch.ones(len(rows_cm)),
        size=(num_api, edge_id)
    ).to_dense()

    return H_cs, H_cm  
  

def build_co_tag_hypergraph_dense(df, column_name):
    """
    构建 Co-Tag 超图（稠密形式），每个标签对应一个超边。

    参数:
        df: 包含标签字段的 DataFrame（如 api_df）
        column_name: 标签字段名（如 'ApiTags'），每行为字符串格式的 list

    返回:
        H_tag: 稠密 incidence 矩阵 (num_nodes × num_tags)
    """
    num_nodes = len(df)
    tag_to_nodes = {}

    # 遍历每行，解析标签并建立标签到节点的映射
    for idx, row in df.iterrows():
        try:
            tags = ast.literal_eval(row[column_name])
        except Exception as e:
            continue  # 跳过格式错误的行

        for tag in tags:
            tag_to_nodes.setdefault(tag, []).append(idx)

    # print(tag_to_nodes)

    # 构建稠密矩阵所需的 row/col 索引
    rows, cols = [], []
    for edge_id, node_list in enumerate(tag_to_nodes.values()):
        for node in node_list:
            rows.append(node)
            cols.append(edge_id)

    # 构造稠密的超图矩阵
    H_tag = torch.sparse_coo_tensor(
        indices=[rows, cols],
        values=torch.ones(len(rows)),
        size=(num_nodes, len(tag_to_nodes))
    ).to_dense()

    return H_tag



def build_co_provider_hypergraph_dense(df, column_name="ApiProvider"):
    """
    构建 Co-Provider 超图（稠密形式），每个 provider 构成一个超边。

    参数:
        df: 包含提供者字段的 DataFrame（如 api_df）
        column_name: 提供者字段名（如 'ApiProvider'）

    返回:
        H_provider: 稠密 incidence 矩阵 (num_nodes × num_providers)
    """
    num_nodes = len(df)
    provider_to_nodes = {}

    for idx, row in df.iterrows():
        provider = str(row[column_name]).strip()
        if provider:
            provider_to_nodes.setdefault(provider, []).append(idx)

    rows, cols = [], []
    for edge_id, node_list in enumerate(provider_to_nodes.values()):
        for node in node_list:
            rows.append(node)
            cols.append(edge_id)

    H_provider = torch.sparse_coo_tensor(
        indices=[rows, cols],
        values=torch.ones(len(rows)),
        size=(num_nodes, len(provider_to_nodes))
    ).to_dense()

    return H_provider


def build_co_description_hypergraph_by_id(json_path, df, top_k=3,the_type="mashup"):
    """
    根据 ID 顺序从 JSON 嵌入文件构建描述相似超图。

    参数:
        json_path: str, 路径，格式为 {name: [embedding]}
        df: pd.DataFrame，包含 'ID' 和 'MashupName' 两列
        top_k: int，每个 mashup 连接 top_k 个最相似节点构成超边

    返回:
        H_desc: 稠密 incidence 矩阵 (num_mashup × num_hyperedges)，顺序与 ID 对齐
    """
    # 读取 BERT 嵌入
    with open(json_path, 'r') as f:
        name_to_emb = json.load(f)

    # 1. 以 ID 升序排序
    df = df.sort_values("ID")
    if the_type=='mashup':
        names_in_order = df["MashupName"].tolist()
    elif the_type=='api':
        names_in_order = df["ApiName"].tolist()

    # 2. 获取嵌入列表
    embeddings = []
    for name in names_in_order:
        if name not in name_to_emb:
            raise ValueError(f"BERT embedding not found for mashup: {name}")
        emb = torch.tensor(name_to_emb[name], dtype=torch.float32)
        embeddings.append(emb)

    embeddings = torch.stack(embeddings, dim=0)  # (N, D)
    num_nodes = embeddings.size(0)

    # 3. 计算余弦相似度
    norm_emb = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(norm_emb, norm_emb.T)

    # 4. 构建超边（每个节点一条，连接自己 + top_k 相似节点）
    rows, cols = [], []
    edge_id = 0
    for i in range(num_nodes):
        sim_matrix[i, i] = -1
        topk = torch.topk(sim_matrix[i], k=top_k).indices.tolist()
        hyperedge_nodes = [i] + topk
        for node in hyperedge_nodes:
            rows.append(node)
            cols.append(edge_id)
        edge_id += 1

    H_desc = torch.sparse_coo_tensor(
        indices=[rows, cols],
        values=torch.ones(len(rows)),
        size=(num_nodes, edge_id)
    ).to_dense()

    return H_desc