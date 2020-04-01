# encoding:utf-8
import pickle as pkl
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp


def get_matrixM(path):
    res = {}
    user_list = []
    item_list = []
    with open(path) as f: 
        line = f.readline()
        #按行读数据
        while line:
            eles = line.split(',')
            #源数据是csv格式的
            if eles[0] not in res.keys():
                #如果当前学号在之前未被记录过
                res[eles[0]] = {}
                user_list.append(eles[0])
            if eles[1] not in item_list:
                #如果当前课程在之前未被记录过
                item_list.append(eles[1])
            try:
                #20 res[eles[0]][eles[1]] = int(float(eles[2])/10)
                res[eles[0]][eles[1]] = int(float(eles[2]) / 5)
            except:
                pass
            line = f.readline()

    data=pd.DataFrame(res).T # 成绩矩阵，行表示用户，列表示课程
    data=pd.DataFrame.fillna(data,0)# 用０填充nan

    c=np.array(data.columns).reshape([data.shape[1],1],)
    r=np.array(data.index).reshape([data.shape[0],1])
    r_c=np.vstack((r,c))
    # np.savetxt('./data/meta.txt', r_c)
    print('r_c={0}'.format(r_c))

    M, N = data.shape
    '''
    data=np.array(data)
    matrix_all=[] # 存储１－１０个级别对应的0/1矩阵
    for i in range(10):
        matrix=np.zeros([M,N])
        matrix_all.append(matrix)
    for row in range(M):
        for cow in range(N):
            if not np.isnan(data[row][cow]):
                matrix_all[int(data[row][cow]-1)][row][cow]=1  #在对应值的对应级别上修改矩阵
    '''
    one_hot_data = np.array([one for one in range(N+M)])  
    #得到成绩矩阵的
    data_onehot = pd.get_dummies(one_hot_data)
    return data_onehot,data

def get_adj_01(adj):
    M, N = adj.shape
    matrix_all = []  # 存储１－１０个级别对应的0/1矩阵
    #20
    for i in range(20):
        matrix = np.zeros([M, N])
        matrix_all.append(matrix)
    for row in range(M):
        for cow in range(N):
            if int(adj[row][cow]) != 0:
            #if not np.isnan(adj[row][cow]):
                matrix_all[int(adj[row][cow] - 1)][row][cow] = 1  # 在对应值的对应级别上修改矩阵

    return matrix_all

def get_marix_combine_matixT(matrix):# 矩阵matrix和其转置矩阵分在处在对角位置组成对称的邻接矩阵
    '   0    A.T  '

    '   A     0   '
    matrixT = matrix.T
    zero_h = np.zeros([matrix.shape[0], matrix.shape[0]])
    matrix = np.hstack((zero_h, matrix))
    zero_v = np.zeros([matrixT.shape[0], matrixT.shape[0]])
    matrixT = np.hstack((matrixT, zero_v))
    new = np.vstack((matrix, matrixT))
    return new

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index



if __name__=='__main__':
    print()
    # data, adj = getData('./data/B46_154611_chengji.csv')
    #adj = sp.coo_matrix(adj)
    #print(adj.shape)

