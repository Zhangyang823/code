#!/usr/bin/python
#-*- coding: utf-8
import numpy as np
import scipy.sparse as sp


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        #判断sparse_mx是否是稀疏阵，若不是先转化为coo类型稀疏阵（便于存储）
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
    #coords是取边
'''
>>> row  = np.array([0, 3, 1, 0])
>>> col  = np.array([0, 3, 1, 2])
>>> data = np.array([4, 5, 7, 9])
>>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
array([[4, 0, 9, 0],
       [0, 7, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 5]])
'''

def preprocess_graph(adj):
    #对于adj矩阵的标准化
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    #adj矩阵加上单位矩阵
    rowsum = np.array(adj_.sum(1))
    #对列求和，存储在ndarry中，再求取平方根倒数
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    #则标准化后的adj矩阵为：L*D^(-1/2 T)*D^(-1/2)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # 参数是原矩阵 标准化后矩阵 
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    # {placeholders['support'][i]: support[i] for i in range(len(support))}
    feed_dict.update({placeholders['adj'][i]: adj_normalized[i] for i in range(len(adj_normalized))})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict


def mask_test_edges(adj):
    # 抽取10%数据作为测试集
    # 获得测试集的方式是shuffle的
    # TODO: Clean up.

    # adj减去单位阵
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    #去0方便存储
    adj.eliminate_zeros()
    # 若对角元素为0的话
    assert np.diag(adj.todense()).sum() == 0
    #取adj上三角
    adj_triu = sp.triu(adj)
    #获得上三角矩阵的value等等
    adj_tuple = sparse_to_tuple(adj_triu)

    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    #划分验证集和测试集
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))
    #print(num_test==2*num_val)   
    all_edge_idx = list(range(edges.shape[0]))  
    np.random.shuffle(all_edge_idx)
    #取索引再打乱
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))
    #构建测试集
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    #构建验证集，操作类似
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            #如果验证集为空
            continue
        if ismember([idx_i, idx_j], train_edges):
            #若验证集和训练集有交集，则不将重复的边加入验证集
            #以下类似
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

   

    
    data = [adj.toarray()[i[0], i[1]] for i in train_edges]
    


    # 重建成绩矩阵
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


