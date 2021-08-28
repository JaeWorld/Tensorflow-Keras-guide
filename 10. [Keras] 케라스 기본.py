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
# # 케라스(Keras)란? 
# 딥러닝을 쉽게 구현할 수 있는 상위 레벨의 인터페이스


# ------ 1. 전처리 ------

# --- Tokenizer ---
# 문장 토큰화에 사용
# 주요 메서드: fit_on_texts(), texts_to_sequences(), word_index
from tensorflow.keras.preprocessing.text import Tokenizer

t = Tokenizer()
fit_text = "The earth is an awesome place to live"
t.fit_on_texts([fit_text])

test_text = "The earth is an great place to live"
sequences = t.texts_to_sequences([test_text])[0]

print("sequences: ", sequences)
print("word_index: ", t.word_index)

# --- pad_sequence() ---
# 모든 샘플의 길이를 동일하게 맞춰주는 패딩 작업을 수행
# 보통 숫자 0을 넣어서 길이를 맞춤
from tensorflow.keras.preprocessing.sequence import pad_sequences

# pad_sequence(data, maxlen, padding)
# data: 패딩할 데이터, maxlen: 모든 데이터에 대해 정규화 할 길이, padding='pre'이면 앞에 0, 'post'이면 뒤에 0 채움
pad_sequences([[1,2,3], [3,4,5,6], [7,8]], maxlen=3, padding='pre')




# ------ 2. 워드 임베딩 ------
# 워드 임베딩이란? 텍스트 내의 단어들을 밀집 벡터로 만드는 것.

# !! 아래 코드는 실제 동작 코드가 아닌, 의사 코드임!!

# --- Embedding() ---
# 단어를 밀집 벡터로 만듦
# 정수 인코딩 된 단어들을 입력받아 임베딩을 수행
# 모델의 첫번째 레이어로만 사용 가능
# text=[['Hope', 'to', 'see', 'you', 'soon'],['Nice', 'to', 'see', 'you', 'again']]

# t = Tokenizer()
# t.fit_on_texts(text)
# text = t.texts_to_sequences(text)
# print(text)

# # Embedding(input_dim, output_dim, input_length=5)
# # input_dim: 단어의 개수, output_dim: 임베딩한 후의 벡터 크기, input_length: 각 입력 시퀀스의 길이
# Embedding(7, 2, input_length=5)




# ------ 3. 모델링 ------

# --- Sequential() ---
# 인공신경망에서 여러 층을 구성하기 위해 사용됨.
# Sequential()로 모델을 선언 한 뒤, model.add()로 층을 단계적으로 추가
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
# model.add(모델)

# --- Dense() ---
# Fully connected layer 을 추가
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dense(output_dim, input_dim, activation)
# output_dim: 출력 뉴런의 수, input_dim: 입력 뉴런의 수, activation: 활성화 함수 (linear, sigmoid, relu, softmax)
# -> 입력층의 뉴런 수가 3, 출력층의 뉴런 수가 1인 신경망
model = Sequential()
model.add(Dense(1, input_dim=3, activation='relu'))

# -> 입력층의 뉴런 수가 4, 은닉층의 뉴런 수가 8, 출력층의 뉴런 수가 1인 신경망
# 두번째 Dense()에는 input_dim 필요 없음. 이전층의 뉴런 수가 8개라는 것을 이미 알고 있기 때문.
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# --- summary() ---
# 모델의 정보를 요약해서 보여줌
model.summary()




# ------ 4. 컴파일과 훈련 ------

# --- compile() ---
# 모델을 기계가 이해할 수 있도록 컴파일함.
# 오차 함수, 최적화 방법, 메트릭 함수를 선택
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential
max_features = 10000

model = Sequential()
model.add(Embedding(max_features, 32, input_length=5))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer, loss, metrics)
# - optimizer: 훈련 과정을 설정하는 옵티마이저
# - loss: 손실 함수
# - metrics: 평가 지표
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# --- fit() ---
# 모델을 학습
# - 첫번째 인자: 학습 데이터
# - 두번째 인자: 레이블 데이터
# - epochs: 총 훈련 횟수
# - batch_size: 배치 크기

# model.fit(X_train, y_train, epochs=10, batch_size=32)




# ------ 5. 평가와 예측 ------

# --- evaluate() ---
# 학습한 모델에 대한 정확도를 평가

# model.evaluate(X_test, y_test, batch_size=32)

# --- predict() ---
# 입력에 대한 모델의 출력값 확인

# model.predict(X_input, batch_size=32)



# ------ 6. 모델의 저장과 로드 ------

# --- save() ---
# 모델을 hdf5 파일에 저장
# model.save("model_name.h5")

# --- load_model() ---
# 저장해둔 모델 로드
# from tensorflow.keras.models import load_model
# model = load_model("model_name.h5")
# -


