# encoding:utf-8
#这个文件定义的是“图卷积”这一基本操作，以及为了降低计算量而设计的切比雪夫展开
from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    #用于无标签的无向图的图卷积层
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class GraphConvolution_Chebyshev5(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    #用于有标签的无向图的图卷积层
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution_Chebyshev5, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        for i in range(0,1):
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    #用于稀疏矩阵的图卷积
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class Xunying_InnerProductDecoder(Layer):
    
    #成绩预测模型 encoder 的输出 Z 作为 decoder 的输入，decoder 输出 M_hat
    #M_hat包含预测边的权重，即学生对应课程的成绩
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(Xunying_InnerProductDecoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        outputs = []
        for i in range(10):
            intputs_i = tf.nn.dropout(inputs, 1 - self.dropout)
            W = weight_variable_glorot(self.input_dim, self.input_dim, name="decoder_weights" + str(i))
            x = tf.transpose(intputs_i)
            x = tf.matmul(tf.matmul(intputs_i, W), x)
            
            outputs.append(self.act(x))
        outputs = tf.stack(outputs, axis=2)  

        #################
        # decoder
        outputs = tf.reshape(outputs, [-1, 10])
        logits_output = tf.nn.softmax(outputs) 
        #Softmax 层神经元个数为 R=10。成绩预测模型的输出 M_hat 和模型的输入 M 对应，预测结果即
        #为学生对应课程的成绩
        r_weight = tf.reshape(tf.constant(list(range(1, 11)),dtype=tf.float32), [10,1])  # (n*n,1)
        n = inputs.get_shape().as_list()[0]
        M_hat = tf.matmul(logits_output, r_weight)
        M_hat = tf.reshape(M_hat, [n, n]) 

       
        ################
        return M_hat,tf.reshape(logits_output,[n,n,10])


