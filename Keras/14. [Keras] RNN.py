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
# ------ 케라스로 RNN 구현 ------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN

# batch_size = 한 번에 학습하는 데이터의 개수
# hidden_size = 은닉 상태의 크기를 정의. 메모리 셀이 다음 시점의 메모리 셀과 출력층으로 보내는 값의 크기(output_dim)와도 동일. RNN의 용량(capacity)을 늘린다고 보면 되며, 중소형 모델의 경우 보통 128, 256, 512, 1024 등의 값을 가진다.
# timesteps = 입력 시퀀스의 길이(input_length)라고 표현하기도 함. 시점의 수.
# input_dim = 입력의 크기.

# RNN 층은 (batch_size, timesteps, input_dim) 크기의 3D 텐서를 입력받아
# (batch_size, output_dim) 크기의 2D 텐서를 반환 (최종 시점의 은닉 상태만 리턴하는 경우)
# (batch_size, timesteps, output_dim) 크기의 3D 텐서를 반환 (각 시점의 은닉 상태를 모두 리턴하는 경우)

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2, 10)))
# model.add(SimpleRNN(3, input_length=2, input_dim=10)) 와 동일.
model.summary()

# -

model = Sequential()
model.add(SimpleRNN(3, input_shape=(2, 10)))
model.summary()


# return_sequences=True 속성으로 각 시점의 은닉상태값들을 모아 전체 시퀀스 리턴 가능
model = Sequential()
model.add(SimpleRNN(3, batch_input_shape=(8, 2, 10), return_sequences=True))
model.summary()


# +
# ------ 파이썬으로 RNN 구현 ------

'''
의사 코드

hidden_state_t = 0 
for input_t in input_length:
    output_t = tanh(input_t, hidden_state_t)
    hidden_state_t = output_t
    
'''


# +
import numpy as np

timesteps = 10 # 시점의 수, NLP에서는 보통 문장의 길이가 된다.
input_dim = 4 # 입력의 차원, NLP에서는 보통 단어 벡터의 차원이 된다.
hidden_size = 8 # 은닉 상태의 크기, 메모리 셀의 용량이다.

inputs = np.random.random((timesteps, input_dim))
hidden_state_t = np.zeros((hidden_size,))

Wx = np.random.random((hidden_size, input_dim)) # 입력에 대한 가중치
Wh = np.random.random((hidden_size, hidden_size)) # 은닉 상태에 대한 가중치
b = np.random.random((hidden_size,)) # 편향

print(np.shape(Wx))
print(np.shape(Wh))
print(np.shape(b))

# +
total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t))
    print(np.shape(total_hidden_states))
    hidden_state_t = output_t
    
total_hidden_states = np.stack(total_hidden_states, axis=0)

print(total_hidden_states)

# -

# ------ 깊은 순환 신경망 ------
# 다수의 은닉층을 가지는 경우
model = Sequential()
model.add(SimpleRNN(hidden_size, return_sequences=True))
model.add(SimpleRNN(hidden_size, return_sequences=True))


# +
# ------ 양방향 순환 신경망 ------
# 이전 시점의 데이터뿐만 아니라, 이후 시점의 데이터도 힌트로 활용하기 위함
# 두 개의 메모디 셀을 사용. 각각 앞 시점/뒤 시점의 은닉 상태를 전달.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Bidirectional

model = Sequential()
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True), input_shape=(timesteps, input_dim)))
# -

# 은닉층 추가한 양방향 RNN
model = Sequential()
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True), input_shape=(timesteps, input_dim)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences=True)))

# +
from tensorflow.keras.layers import Embedding, Dense

model = Sequential()
model.add(Embedding(5000, 100, input_length=30))
model.add(SimpleRNN(128))
model.add(Dense(1, activation='sigmoid'))

model.summary()
# -


