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

# 자동 미분: tf.GradientTape는 컨텍스트(context) 안에서 실행된 모든 연산을 테이프(tape)에 "기록"합니다. 
# 그 다음 텐서플로는 후진 방식 자동 미분(reverse mode differentiation)을 사용해 테이프에 "기록된" 연산의 그래디언트를 계산합니다.
import tensorflow as tf



# ------그래디언트 테이프------
# 텐서플로우2는 즉시실행모드를 사용하는데, 자동 미분을 하기 위해 필요한 함수, 계산식과 입력 값에 대한 정보를 저장할 기능이 필요하다.
# 따라서 중간 연산 과정을 테이프에 기록해주는 GradientTape을 제공한다.
# 컨텍스트 안에서 실행된 모든 연산을 테이프에 기록.
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# 입력 x에 대한 z의 도함수 (z을 x에 대해 미분)
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0

# +

x = tf.ones((2, 2))

with tf.GradientTape() as t:
  t.watch(x)
  y = tf.reduce_sum(x)
  z = tf.multiply(y, y)

# 테이프 사용하여 중간값 y에 대한 도함수를 계산. (z을 y에 대해 미분) 
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0


# 동일한 연산에 대해 미분을 계산하려면 persistent=True 속성을 사용.
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    y = x * x
    z = y * y
dz_dx = t.gradient(z, x)
dy_dx = t.gradient(y, x)

print(dz_dx) # 108.0
print(dy_dx) # 6.0

# 테이프에 대한 참조 삭제.
del t




# ------고계도 그레디언트------
# 그래디언트 연산 자체도 미분이 가능.
x = tf.Variable(1.0)

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    dy_dx = t2.gradient(y, x)
d2y_dx2 = t.gradient(dy_dx, x)

print(dy_dx)
print(d2y_dx2)


# 8.0 = 2.0 * x 방정식에서 변수 x의 값을 찾아보기.
a, y = tf.constant(2.0), tf.constant(8.0)
x = tf.Variable(10.0)

def train_func():
    with tf.GradientTape() as t:
        loss = tf.math.abs(a * x - y)
    
    dx = t.gradient(loss, x)
    print(f'x = {x.numpy()}, dx = {dx:.2f}')
    
    # x을 x-dx로 갱신
    x.assign(x - dx)
    
for i in range(4):
    train_func()


