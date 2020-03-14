#!/usr/bin/python
#-*- coding: utf-8
from layers import GraphConvolution, GraphConvolutionSparse, Xunying_InnerProductDecoder,GraphConvolution_Chebyshev5
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.z_mean = tf.Variable(tf.zeros([self.input_dim, FLAGS.hidden2]), name='zmean')
        for i in range(10):
            self.hidden1_i = GraphConvolutionSparse(input_dim=self.input_dim,
                                                    output_dim=FLAGS.hidden1,
                                                    adj=self.adj[i],
                                                    features_nonzero=self.features_nonzero,
                                                    act=tf.nn.relu,
                                                    dropout=self.dropout,
                                                    logging=self.logging)(self.inputs)
            self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                               output_dim=FLAGS.hidden2,
                                               adj=self.adj[i],
                                               act=lambda x: x,
                                               dropout=self.dropout,
                                               logging=self.logging)(self.hidden1_i)

            self.z_mean = self.embeddings + self.z_mean
        


        self.reconstructions,self.logits_output = Xunying_InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self .z_mean)


class GCNModelVAE(Model):
    #读数据步
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()
        
    

    def _build(self):
        #隐节点分布的均值和标准差取对数，长度是对应的hidden2
        self.z_mean = tf.Variable(tf.zeros([self.input_dim, FLAGS.hidden2]), name='zmean')
        self.z_log_std = tf.Variable(tf.zeros([self.input_dim, FLAGS.hidden2]), name='z_log_std')
        for i in range(10):
            #遍历10个1-0矩阵
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
            #VAE的第一个隐层                                      
                                                  output_dim=FLAGS.hidden1,
                                                  #默认输出维度是64
                                                  adj=self.adj[i],
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging)(self.inputs)
           #GCN的切比雪夫展开用来减少参数量
            self.z_mean1 = GraphConvolution_Chebyshev5(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj[i],
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)
           
            self.z_log_std1 = GraphConvolution_Chebyshev5(input_dim=FLAGS.hidden1,
                                              output_dim=FLAGS.hidden2,
                                              adj=self.adj[i],
                                              act=lambda x: x,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.hidden1)
            #变分自编码器基本操作
            self.z_mean = self.z_mean1 + self.z_mean
            self.z_log_std = self.z_log_std1 + self.z_log_std

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        self.reconstructions, self.logits_output = Xunying_InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)

