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

# 텐서: 일관된 유형을 가진 다차원 배열 (np.array와 같음)
import tensorflow as tf
import numpy as np



# ------텐서 생성------

# scalar(Rank 0)
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
# -

# vector(Rank 1)
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

# matrix(Rank 2)
rank_2_tensor = tf.constant([[1, 2],
                            [3, 4],
                            [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

# Rank 3
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]]
])
print(rank_3_tensor)

# np.array / tensor.numpy 메소드로 텐서를 NumPy 배열로 변환 가능.
np.array(rank_2_tensor)

rank_2_tensor.numpy()




# ------텐서 연산------
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]])

print(tf.add(a, b), '\n')
print(tf.multiply(a, b), '\n')
print(tf.matmul(a, b), '\n')
# -

print(a + b, '\n')
print(a * b, '\n')
print(a @ b, '\n')

# +
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# tf.reduce_max -> 가장 큰 값 찾기
print(tf.reduce_max(c))
# tf.argmax -> 가장 큰 값의 인덱스 찾기
print(tf.argmax(c))
# tf.nn.softmax -> softmax 적용
print(tf.nn.softmax(c))




# ------형상 정보------

rank_4_tensor = tf.zeros([3, 2, 4, 5])

print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())




# ------인덱싱------

# 단일 축 인덱싱 : 표준 파이썬 인덱싱 규칙을 따름.
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())

print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())
# -

print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

# 다축 인덱싱 : 여러 인덱스를 전달하여 인덱싱.
print(rank_2_tensor.numpy())

print("[1, 1] 위치의 값: ", rank_2_tensor[1, 1].numpy())
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

# 3축 텐서 인덱싱
print(rank_3_tensor[:, :, 4])




# ------형상 조작하기------

var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print(var_x.shape)
# -

# 파이썬 리스트 형식으로 변환
print(var_x.shape.as_list())

# reshape로 새로운 형상으로 변환. 리스트로 변환할 모양을 명시.
reshaped = tf.reshape(var_x, [1, 3])
print(var_x.shape)
print(reshaped.shape)

print(rank_3_tensor)

# shape을 '-1'로 지정하면 모양에 맞게 알아서 맞춰줌.
print(tf.reshape(rank_3_tensor, [-1]))

# 일반적으로 tf.reshape의 합리적인 용도는 인접한 축을 결합하거나 분할하는 것.
# 3x2x5 텐서의 경우, (3x2)x5 또는 3x(2x5)로 재구성하는 것이 합리적
print(tf.reshape(rank_3_tensor, [3*2, 5]), '\n')
print(tf.reshape(rank_3_tensor, [3, -1]))




# ------브로드캐스팅------

x = tf.constant([1, 2, 3])
y = tf.constant([2])
z = tf.constant([2, 2, 2])

print(tf.multiply(x, 2))
print(x * y)
print(x * z)

# +
x = tf.reshape(x, [3, 1])
y = tf.range(1, 5)

print(x, '\n')
print(y, '\n')
print(tf.multiply(x, y))

# +
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)




# ------비정형(Ragged) 텐서------

ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]
]
# -

# 비정형 텐서를 정규 텐서로 표현 할 수 없음. -> tf.ragged.constant 사용.
try:
    tensor = tf.constant(ragged_list)
except Exception as e:
    print(f"{type(e).__name__}: {e}")

ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)

print(ragged_tensor.shape)




# ------문자열 텐서------
# 문자열은 Python 문자열과 같은 방식으로 인덱싱 불가. 문자열의 길이는 텐서의 차원이 아님.

# 스칼라 문자열 텐서
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)

# 문자열 벡터
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
print(tensor_of_strings)
# 'b'접두사는 dtype이 유니코드 문자열이 아니라 바이트 문자열임을 나타냄.

# 유니코드 문자를 전달하면 UTF-8로 인코딩됨.
tf.constant("🥳👍")

# tf.strings.split
print(tf.strings.split(scalar_string_tensor, sep=" "))
print(tf.strings.split(tensor_of_strings))

# tf.strings.to_number
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, sep=" ")))




# ------희소 텐서------

sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                      values=[1, 2],
                                      dense_shape=[3, 4])
print(sparse_tensor)
print(tf.sparse.to_dense(sparse_tensor))


