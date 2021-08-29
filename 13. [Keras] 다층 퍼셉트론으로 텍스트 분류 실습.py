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
# 다층 퍼셉트론이란? 
# 단층 퍼셉트론 형태에서 은닉층이 1개 이상 추가된 신경망. 
# 피드 포워드 신경망의 가장 기본적인 형태 (입력층에서 출력층으로, 오직 한 방향으로만 연산 수행)

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']

t = Tokenizer()
t.fit_on_texts(texts)
print(t.word_index)


# +
# texts_to_matrix()
# 입력된 텍스트 데이터로부터 행렬을 만드는 함수
# 4개의 mode 지원 ('binary', 'count', 'freq', 'tfidf')

# mode='count' : 등장 횟수 카운트
print(t.texts_to_matrix(texts, mode='count'))
# -

# mode='binary' : 단어가 존재하면 1, 아니면 0
print(t.texts_to_matrix(texts, mode='binary'))

# mode='tfidf' : tfidf 행렬
print(t.texts_to_matrix(texts, mode='tfidf').round(2))

# mode='freq' : 각 단어의 등장 횟수 / 각 문서에서 등장한 모든 단어 개수의 총 합
print(t.texts_to_matrix(texts, mode='freq').round(2))


# +
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
# %matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

newsdata = fetch_20newsgroups(subset='train')
print(newsdata.keys())

print('훈련용 샘플의 개수: ', len(newsdata.data))

print('총 주제의 개수 : {}'.format(len(newsdata.target_names)))

print(newsdata.target_names)

print('첫번째 샘플의 레이블 : {}'.format(newsdata.target[0]))

print('7번 레이블이 의미하는 주제 : {}'.format(newsdata.target_names[7]))
# -

print(newsdata.data[0])

data = pd.DataFrame(newsdata.data, columns=['email'])
data['target'] = pd.Series(newsdata.target)
data[:5]

data.info()

print('중복을 제외한 샘플의 수 : {}'.format(data['email'].nunique()))
print('중복을 제외한 주제의 수 : {}'.format(data['target'].nunique()))

data['target'].value_counts().plot(kind='bar')

print(data.groupby('target').size().reset_index(name='count'))

# +
newsdata_test = fetch_20newsgroups(subset='test', shuffle=True)
train_email = data['email']
train_label = data['target']
test_email = newsdata_test['data']
test_label = newsdata_test['target']

max_words = 10000
num_classes = 20

# -

def prepare_data(train_data, test_data, mode):
    t = Tokenizer(num_words=max_words)
    t.fit_on_texts(train_data)
    X_train = t.texts_to_matrix(train_data, mode=mode)
    X_test = t.texts_to_matrix(test_data, mode=mode)
    return X_train, X_test, t.index_word


X_train, X_test, index_to_word = prepare_data(train_email, test_email, 'binary')
y_train = to_categorical(train_label, num_classes)
y_test = to_categorical(test_label, num_classes)

# max_words을 10000으로 지정해주어서 n x 10000 크기의 행렬로 변환
# 단어의 정수 인덱스는 1부터 시작하므로, 0번 인덱스는 사용 x. 따라서 빈도수 상위 9999개 단어가 표현됨.
print('훈련 샘플 본문의 크기 : {}'.format(X_train.shape))
print('훈련 샘플 레이블의 크기 : {}'.format(y_train.shape))
print('테스트 샘플 본문의 크기 : {}'.format(X_test.shape))
print('테스트 샘플 레이블의 크기 : {}'.format(y_test.shape))

print('빈도수 상위 1번 단어 : {}'.format(index_to_word[1]))
print('빈도수 상위 9999번 단어 : {}'.format(index_to_word[9999]))

# +
# 다층 퍼셉트론 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def fit_and_evaluate(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(256, input_dim=max_words, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1, validation_split=0.1)
    score = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
    return score[1]


# +
modes = ['binary', 'count', 'tfidf', 'freq']

for mode in modes:
    X_train, X_test, _ = prepare_data(train_email, test_email, mode)
    score = fit_and_evaluate(X_train, y_train, X_test, y_test)
    print(mode+' 모드의 테스트 정확도: ', score)
# -


