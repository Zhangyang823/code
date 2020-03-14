#!/usr/bin/python
#-*- coding: utf-8
from __future__ import division
from __future__ import print_function

import time
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import confusion_matrix,auc
from optimizer import OptimizerAE, OptimizerVAE
from input_data import get_adj_01,get_matrixM,get_marix_combine_matixT
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# 参数默认值
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
model_str = FLAGS.model


# 读取源数据
data_onehot,adj= get_matrixM('./data/2013-2016_1_4611_chengji.csv')
adj=get_marix_combine_matixT(adj)
adj=sp.csr_matrix(adj)

# 用scipy的sparse包存储和处理稀疏矩阵
features=data_onehot   
features=sp.lil_matrix(features)

# 邻接矩阵自减去对角线元素
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
#
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj_Rs = get_adj_01(adj_train.toarray())
adj=adj_train
adj_R=[sp.csr_matrix(adj_one) for adj_one in adj_Rs]

#若不使用采样得来的特征，则特征矩阵使用单位矩阵
if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  

# 预处理，主要内容写在preprocess模块里了
adj_R_norm = [preprocess_graph(one_adj) for one_adj in adj_R]

# placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': [tf.sparse_placeholder(tf.float32) for _ in range(10)],
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# 建立模型，从传统自编码器和vae中二选一
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# 优化器，作为baseline不用改动了
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':

        n,_,_=model.logits_output.get_shape().as_list()
        tensor_list=[]
        for adj_channel in placeholders['adj']:
            channel_tensor=tf.reshape(tf.sparse_tensor_to_dense(adj_channel, validate_indices=False), [n,n])
            tensor_list.append(channel_tensor)

        adj_tensor=tf.stack(tensor_list,axis=2)

        opt = OptimizerAE(preds=model.logits_output,
                          labels=adj_tensor,
                          pos_weight=pos_weight,
                          norm=norm)

    elif model_str == 'gcn_vae':
        n, _, _ = model.logits_output.get_shape().as_list()
        tensor_list = []
        for adj_channel in placeholders['adj']:
            channel_tensor = tf.reshape(tf.sparse_tensor_to_dense(adj_channel, validate_indices=False), [n, n])
            tensor_list.append(channel_tensor)

        adj_tensor = tf.stack(tensor_list, axis=2)

        opt = OptimizerVAE(preds=model.logits_output,
                           labels=adj_tensor,
                           pos_weight=pos_weight,
                           model=model,
                           num_nodes=n,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []

# 给定混淆矩阵M，计算查全和查准率。

def get_precision_recall_score_confusion(M):
    n = len(M)
    precision=[]
    recall=[]
    for i in range(len(M[0])):
        
        rowsum, colsum = sum(M[i]), sum(M[r][i] for r in range(n))
        try:
            
            precision.append(M[i][i] / float(colsum))
            recall.append(M[i][i] / float(rowsum))
        except ZeroDivisionError:
            precision.append(0)
            recall.append(0)
    return np.mean([i for i in precision if not np.isnan(i)]),np.mean([i for i in recall if not np.isnan(i)])

def get_roc_score_r(edges_pos, edges_neg, emb=None):
    #此函数用来计算验证集和测试集的各项评估误差
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        adj_rec = sess.run(model.reconstructions,feed_dict=feed_dict)

    
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])
        pos.append(adj_orig[e[0], e[1]])
        
    preds_neg = []
    neg = []
    for e in edges_neg:
         
        preds_neg.append(adj_rec[e[0], e[1]])
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    #labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    labels_all = np.hstack([pos, neg])
    loss_rmse = np.sqrt(np.mean((np.floor(np.array(pos))-np.array(preds))**2))  
    loss_mse = np.mean((np.floor(np.array(pos))-np.array(preds))**2)
    loss_mae = np.mean(np.abs(np.floor(np.array(pos))-np.array(preds)))

    #print('loss_mean:{0},loss_mse:{1},loss_mae:{2}'.format(loss_rmse,loss_mse,loss_mae))
    return loss_rmse,loss_mse,loss_mae


def get_roc_score(edges_pos, edges_neg, emb=None):
    #得到混淆矩阵
    adj_rec=sess.run(model.reconstructions,feed_dict=feed_dict)
    preds = []
    pos = []
    for e in edges_pos:
        if not np.isnan(adj_orig[e[0], e[1]]):
            preds.append(adj_rec[e[0], e[1]])
            pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        if not np.isnan(adj_orig[e[0], e[1]]):
            preds_neg.append(adj_rec[e[0], e[1]])
            neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])

    labels_all = np.hstack([pos, neg])
    ncorrects = sum(preds_all == labels_all)
    #print('ncorrects:'.format(ncorrects))
    '''
    ncorrects = sum(predictions == labels)
    accuracy = 100 * sklearn.metrics.accuracy_score(labels, predictions)
    f1 = 100 * sklearn.metrics.f1_score(labels, predictions, average='weighted')
     '''

    #roc_score = roc_auc_score(labels_all, preds_all)
    #ap_score = average_precision_score(labels_all, preds_all)
    preds_all=[round(one*10) for one in preds_all]
    confusion=confusion_matrix(labels_all,preds_all)

    precision,recall=get_precision_recall_score_confusion(confusion)
    #得到查全、查准率
   
    return precision,recall


cost_val = []
acc_val = []


val_roc_score = []
avg_cost_list=[]
adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
time_list=[]
# Train model
for epoch in range(FLAGS.epochs):
    #计时
    t = time.time()
    feed_dict = construct_feed_dict(adj_R_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

     
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    roc_curr, ap_curr,_ = get_roc_score_r(val_edges, val_edges_false)
    
    #val_roc_score.append(roc_curr) 取消注释
    
    time_list.append(time.time() - t)
    avg_cost_list.append(avg_cost)
    test_rmse,test_mse, test_mae = get_roc_score_r(test_edges, test_edges_false)
    'train_rmse,any1,any2 = get_roc_score_r( train_edges,)'
    val_rmse,val_mse, val_mae = get_roc_score_r(val_edges, val_edges_false) 
    print("Epoch:", '%04d' % (epoch + 1), 
          "Train_loss=", "{:.5f}".format(avg_cost),
          #"train_acc=", "{:.5f}".format(avg_accuracy),    #训练集的正确率
          #"Val_RMSE1=", "{:.5f}".format(val_roc_score[-1]),
          #"Val_RMSE2=", "{:.5f}".format(val_rmse),
          "Val_RMSE= {0},Val_MSE= {1},Val_MAE= {2}".format(val_rmse,val_mse, val_mae),
          "Test_RMSE= {0},Test_MSE= {1},Test_MAE= {2}".format(test_rmse,test_mse, test_mae),
          "time=", "{:.5f}".format(time.time() - t),
          "Model=",model_str,
          '\n')
    #训练模型是靠 减少train_set的loss
    #而model的评估指标是val_set和test_set的误差rmse mse mae

print("Optimization Finished!")

#print(val_roc_score)  此项输出是待选的
