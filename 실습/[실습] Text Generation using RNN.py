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
# ------ 1. 데이터 전처리 ------

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

text = """경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고와야 오는 말이 곱다\n"""

t = Tokenizer()
t.fit_on_texts([text])
vocab_size = len(t.word_index) + 1

print('단어 집합의 크기: %d' % vocab_size)
# -

print(t.word_index)

# +
sequences = []
for line in text.split('\n'):
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)
        
print('학습에 사용할 샘플의 개수: %d' % len(sequences))
# -

# 아직 레이블로 사용될 단어를 분리하지 않은 훈련 데이터
# 맨 우측에 있는 단어만 레이블로 분리
print(sequences)


# 전체 샘플에 대해 길이를 맞춰준다. 가장 긴 샘플의 길이를 기준으로.
max_len = max(len(l) for l in sequences)
print('샘플의 최대 길이: ', max_len)

# 패딩 진행
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences)


# +
# 마지막 단어만 레이블로 분리.
# numpy 사용
sequences = np.array(sequences)
X = sequences[:, :-1]
y = sequences[:, -1]

print(X)
print(y)

# -

# RNN 모델에 훈련시키기 위해 레이블을 원-핫 인코딩 수행
y = to_categorical(y)
print(y)


# +
# ------ 2. 모델 설계하기 ------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
model.add(SimpleRNN(32))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)

# -

model.predict_classes([10, 1])


def sentence_generation(model, t, current_word, n):
    init_word = current_word
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0]
        encoded = pad_sequences([encoded], maxlen=5, padding='pre')
        print(encoded)
        result = model.predict_classes(encoded, verbose=0)
        for word, index in t.word_index.items():
            if index == result:
                current_word = current_word + ' ' + word
                break
        sentence = sentence + ' ' + word
    sentence = init_word + sentence
    return sentence


print(sentence_generation(model, t, '그의', 2))

print(sentence_generation(model, t, '경마장에', 4))


