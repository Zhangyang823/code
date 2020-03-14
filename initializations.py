#!/usr/bin/python
#-*- coding: utf-8
import tensorflow as tf
import numpy as np

def weight_variable_glorot(input_dim, output_dim, name=""):
    """实现Glorot & Bengio (AISTATS 2010)
   提出的权重初始化
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)
