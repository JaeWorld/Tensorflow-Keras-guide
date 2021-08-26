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
# ------ 케라스 함수형 API ------

# Sequential API는 복잡한 모델을 만드는데는 한계가 있음. 
# 복잡한 모델을 생성하기 위해서는 함수형 API 사용해야함.


# +
# ----- Sequential API 모델 ------
# 간단하긴 하지만 복잡한 신경망 구현 불가
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))


# +
# ------ Functional API 모델 ------
# 함수형 API는 함수로서 정의.
# 각 함수를 조합하기 위한 연산자를 제공

'''
Functional API 모델 생성 과정
1. Input() 함수에 입력 크기 정의
2. 이전층을 다음층 함수의 입력으로 사용하고 변수에 할당
3. Model() 함수에 입력과 출력을 정의
'''

# 함수형 API에서는 Sequential와는 달리, 입력층을 정의해주어야 한다.
# 입력 데이터의 크기를 인자로 주어 입력층 정의 (Input layer)
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# 입력층 정의
inputs = Input(shape=(10,))

# 은닉층, 출력층 정의
hidden1 = Dense(64, activation='relu')(inputs)
hidden2 = Dense(64, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)

# Model에 inputs, outputs 정의하여 하나의 모델로 구성
model = Model(inputs=inputs, outputs=output)


# +
# 생성된 모델로 compile, fit 등 사용 가능
model.compile(optimizer='adam', loss='categorical_crossentropy',
             metrics=['accuracy'])
# model.fit(data, labels)



# +
# --- 선형 회귀 ---
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

X = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y = [11, 22, 33, 44, 53, 66, 77, 87, 95]

inputs = Input(shape=(1,))
output = Dense(1, activation='linear')(inputs)
linear_model = Model(inputs=inputs, outputs=output)

sgd = optimizers.SGD(lr=0.01)

linear_model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
linear_model.fit(X, y, batch_size=1, epochs=300, shuffle=False)


# +
# --- 로지스틱 회귀 ---
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(3,))
output = Dense(1, activation='sigmoid')(inputs)
logistic_model = Model(inputs=inputs, outputs=output)


# +
# --- 다중 입력을 받는 모델 ---

# 다중 출력 모델의 예시
# model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])

from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model

# 두 개의 입력층 정의
inputA = Input(shape=(64,))
inputB = Input(shape=(128,))

# 첫번째 입력층으로부터 분기되어 진행되는 신경망 정의
x = Dense(16, activation='relu')(inputA)
x = Dense(8, activation='relu')(x)
x = Model(inputs=inputA, outputs=x)

# 두번째 입력층으로부터 분기되어 진행되는 신경망 정의
y = Dense(64, activation='relu')(inputB)
y = Dense(32, activation='relu')(y)
y = Dense(8, activation='relu')(y)
y = Model(inputs=inputB, outputs=y)

# 두개의 신경망의 출력 연결
result = concatenate([x.output, y.output])

# 연결된 값을 입력으로 받는 밀집층 추가
z = Dense(2, activation='relu')(result)
# 선형 회귀를 위해 activation='linear'
z = Dense(1, activation='linear')(z)

# 결과적으로 이 모델은 두 개의 입력층으로부터 분기되어 진행된 후 마지막에는 하나의 출력을 예측하는 모델이 됨.
model = Model(inputs=[inputA, inputB], outputs=z)


# +
# --- RNN 은닉층 사용 ---
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model

inputs = Input(shape=(50, 1))
lstm_layer = LSTM(10)(inputs)
x = Dense(10, activation='relu')(lstm_layer)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=inputs, outputs=output)

# -


