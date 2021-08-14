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

# 변수: tf.Variable 클래스를 통해 생성, 추적.
import tensorflow as tf


# ------변수 생성------

# tf.Variable -> 변수 생성
my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5+4j, 6+1j])

print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As Numpy: ", my_variable.numpy)

print("A variable:",my_variable)
print("\nViewed as a tensor:", tf.convert_to_tensor(my_variable))
print("\nIndex of highest value:", tf.argmax(my_variable))

# 기존 변수를 reshape하지 않고 새로운 텐서를 생성합니다.
print("\nCopying and reshaping: ", tf.reshape(my_variable, ([1,4])))

# tf.Variable.assign -> 텐서 재할당
a = tf.Variable([2.0, 3.0])
b = tf.Variable(a)
a.assign([5, 6])

print(a.numpy()) #[5. 6.]
print(b.numpy()) #[2. 3.]

print(a.assign_add([2, 3]).numpy()) #[7. 9.]
print(a.assign_sub([7, 9]).numpy()) #[0. 0.]


