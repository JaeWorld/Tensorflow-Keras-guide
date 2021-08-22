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

# +

TRUE_W = 3.0
TRUE_B = 2.0

NUM_EXAMPLES = 1000

x = tf.random.normal(shape=[NUM_EXAMPLES])

noise = tf.random.normal(shape=[NUM_EXAMPLES])

y = x * TRUE_W + TRUE_B + noise


# +
import matplotlib.pyplot as plt

plt.scatter(x, y, c='b')
plt.show()


# +
# 모델 정의하기
class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
        
    def __call__(self, x):
        return self.w * x + self.b
    
model = MyModel()

print('Variables: ', model.variables)

assert model(3.0).numpy() == 15.0

# -

# 손실 함수 정의하기
def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))



# +
plt.scatter(x, y, c='b')
plt.scatter(x, model(x), c='r')
plt.show()

print("Current loss: %1.6f" % loss(y, model(x)).numpy())

# +
# 훈련 루프 정의하기
'''
훈련 루프는 아래 3가지 작업을 반복적으로 수행하는 것으로 구성

- 모델을 통해 입력 배치를 전송하여 출력 생성
- 출력을 출력(또는 레이블)과 비교하여 손실 계산
- 그래디언트 테이프를 사용하여 그래디언트 찾기
- 해당 그래디언트로 변수 최적화
'''

def train(model, x, y, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(y, model(x))
        
    dw, db = t.gradient(current_loss, [model.w, model.b])
    
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)
    


# +
model = MyModel()

Ws, bs = [], []
epochs = range(10)

def training_loop(model, x, y):
    for epoch in epochs:
        train(model, x, y, learning_rate=0.1)
        
        Ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        current_loss = loss(y, model(x))
        
        print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
          (epoch, Ws[-1], bs[-1], current_loss))
        


# +
print("Starting: W=%1.2f b=%1.2f,k loss=%2.5f" %
      (model.w, model.b, loss(y, model(x))))

training_loop(model, x, y)

plt.plot(epochs, Ws, 'r',
        epochs, bs, 'b')

plt.plot([TRUE_W] * len(epochs), 'r--',
        [TRUE_B] * len(epochs), 'b--')

plt.legend(['W', 'b', 'True W', 'True b'])
plt.show()


# +
plt.scatter(x, y, c='b')
plt.scatter(x, model(x), c='r')
plt.show()

print("Current loss: %1.6f" % loss(model(x), y).numpy())


# +
# Keras를 사용한 경우
class MyModelKeras(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)
        
    def call(self, x):
        return self.w * x + self.b
    
keras_model = MyModelKeras()

training_loop(keras_model, x, y)

keras_model.save_weights('my_checkpoint')


# +
keras_model = MyModelKeras()

keras_model.compile(
    run_eagerly=False,
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1),
    loss=tf.keras.losses.mean_squared_error,
)
# -

print(x.shape[0])
keras_model.fit(x, y, epochs=10, batch_size=1000)
