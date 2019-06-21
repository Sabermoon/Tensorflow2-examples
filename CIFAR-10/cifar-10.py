"""
cifar-10数据集，官网链接：https://www.cs.toronto.edu/~kriz/cifar.html

本代码为图像数据增强参考
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.datasets.cifar10 import load_data

# 下载并读取数据集，注意网络质量
(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape,y_train.shape)
# 50000,32,32,3         50000,1

# plt.imshow(x_train[5])
# plt.show()
# plt.imsave('test1.jpg', x_train[5])

# 图像增强，注意根据内存大小调节
for i in range(0, 10):
    batch = x_train.shape[0] // 10
    if i == 0:
        # 图像裁剪
        distorted_train = tf.image.random_crop(x_train[0: batch], size=[batch, 24, 24, 3])
        # 随机旋转（左右）
        distorted_train = tf.image.random_flip_left_right(distorted_train)
        # 随机调整亮度
        distorted_train = tf.image.random_brightness(distorted_train, max_delta=0.25)
        # 随机调整对比度
        distorted_train = tf.image.random_contrast(distorted_train, lower=0.2, upper=1.2)
    else:
        temp = tf.image.random_crop(x_train[i*batch: (i+1)*batch], size=[batch, 24, 24, 3])
        temp = tf.image.random_flip_left_right(temp)
        temp = tf.image.random_brightness(temp, max_delta=0.25)
        temp = tf.image.random_contrast(temp, lower=0.2, upper=0.8)
        distorted_train = tf.concat([distorted_train, temp], axis=0)

# for i in range(distorted_train[10:50].shape[0]):
#     plt.subplot(40//8,8,i+1)
#     plt.imshow(distorted_train[i])
# plt.show()
print(distorted_train.shape)

# 定义网络
model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=(24, 24, 3), filters=64, kernel_size=5, strides=1,
                                                    padding='same', activation=tf.nn.relu),
                             tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                             tf.keras.layers.Conv2D(64, 5, 1, padding='same', activation=tf.nn.relu),
                             tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(384, activation=tf.nn.relu),
                             tf.keras.layers.Dense(192, activation=tf.nn.relu),
                             tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 归一化
distorted_train = tf.cast(distorted_train, tf.float32) / 255.

# 训练模型，注意可用内存大小
# 由于模型变换较大，网络难以训练，需要大量训练，推荐轮次设25以上，准确率才比较高
model.fit(distorted_train, y_train, epochs=25, verbose=2)
print(model.evaluate(distorted_train, y_train)[1])