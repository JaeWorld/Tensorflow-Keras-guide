# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import tensorflow as tf
from datetime import datetime

# %load_ext tensorboard

# +
# TensorFlow에서 Keras 또는 Sonnet과 같은 레이어 및 모델의 상위 수준 구현 대부분은 같은 기본 클래스인 tf.Module를 기반으로 구축됨.
class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name='train_me')
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name='do_not_train_me')
    def __call__(self, x):
        return self.a_variable * x + self.non_trainable_variable
    
simple_module = SimpleModule(name='simple')

simple_module(tf.constant(5.0))

# +

print("trainable variables:", simple_module.trainable_variables)
print("all variables:", simple_module.variables)


# +

class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


# +

class SequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        
        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)
        
    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)
    
my_model = SequentialModule(name='the_model')

print("Model results: ", my_model(tf.constant([[2.0, 2.0, 2.0]])))
# -


