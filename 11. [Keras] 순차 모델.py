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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# +

# Sequential 모델은 각 레이어에 정확히 하나의 입력 텐서와 하나의 출력 텐서가 있는 일반 레이어 스택에 적합.
model = keras.Sequential(
    [
        layers.Dense(2, activation='relu', name='layer1'),
        layers.Dense(3, activation='relu', name='layer2'),
        layers.Dense(4, name='layer3'),
    ]
)

x = tf.ones((3, 3))
y = model(x)


# + active=""
# '''
# Sequential 모델은 다음의 경우에 적합하지 않음.
#
# - 모델에 다중 입력 또는 다중 출력이 있음.
# - 레이어에 다중 입력 또는 다중 출력이 있음.
# - 레이어 공유를 해야 함.
# - 비선형 토폴로지 원함.
# ''' 
#

# +
# ------ Sequential 모델 생성하기 ------
model = keras.Sequential([
    layers.Dense(2, activation='relu'),
    layers.Dense(3, activation='relu'),
    layers.Dense(4),
])

# layer 속성을 통해 속한 레이어에 접근 가능
model.layers

# -

# add 메서드를 통해 점진적으로 모델 작성 가능
model = keras.Sequential()
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(3, activation='relu'))
model.add(layers.Dense(4))


# pop 메서드를 통해 레이어 제거 가능 
model.pop()
print(len(model.layers))


# +
# ------ 미리 입력 형상 지정하기 ------

# Keras의 모든 레이어는 가중치를 만들려면 입력의 형상을 알아야 함.
# 아래와 같이 만들면 입력 형상을 모르므로 가중치가 없음.
layer = layers.Dense(3)
layer.weights

# -

# 입력에서 처음 호출될때 가중치를 만듦. 입력의 형상에 따라 가중치가 달라지기 때문.
x = tf.ones((1, 4))
y = layer(x)
layer.weights


# +
# Sequential 모델도 마찬가지로 입력형상을 모르면 가중치가 없음
model = keras.Sequential([
    layers.Dense(2, activation='relu'),
    layers.Dense(3, activation='relu'),
    layers.Dense(4),
])

# model.weights # <- 실행 불가 (입력 형상을 모르기 때문)
# model.summary() 

x = tf.ones((1, 4))
y = model(x)
print('Number of weights after calling the model: ', len(model.weights))

# -

model.weights

model.summary()


# +
# 첫 번째 레이어에 input_shape 인수를 전달하여 입력 형상을 지정하는게 좋음.
# 빌드된 후 항상 가중치를 가진다. (입력 형상을 알기 때문)
# input_shape 설명 : https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
model = keras.Sequential()
model.add(layers.Dense(2, activation='relu', input_shape=(4,)))

model.summary()
# -

model.weights


# +
# ------ 일반적인 디버깅 워크플로우: add() + summary() ------
# 새로운 Sequential 아키텍처를 구축할 때는 add() 하여 레이어를 점진적으로 쌓고 모델 요약을 자주 인쇄하는 것이 유용
model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))
model.add(layers.Conv2D(32, 5, strides=2, activation='relu'))
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPooling2D(3))

model.summary()

model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))

model.summary()

model.add(layers.GlobalMaxPooling2D())
model.add(layers.Dense(10))


# +
# ------ Sequential 모델을 사용한 특성 추출 ------
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
# -

features


