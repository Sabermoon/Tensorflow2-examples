"""
本样例使用双层卷积网络进行分类，其余部分与softmax大致相同
"""
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# 关闭警告
# import logging
# logging.getLogger('tensorflow').disabled = True

# 读数据
def load_data():
    # 数据集格式为npz，注意修改路径
    with np.load('mnist.npz') as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)

(X_train, y_train), (X_test, y_test) = load_data()

# 预览样本
# plt.imshow(X_train[0], cmap='gray')
# plt.show()

# 转换浮点数
X_train, X_test = X_train/255.0, X_test/255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# 分割训练集、验证集
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=5000, random_state=25)

# softmax回归
model = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding='same',
                                                    input_shape=(28,28,1), activation=tf.nn.relu),
                             tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
                             tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same',
                                                    activation=tf.nn.relu),
                             tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                             tf.keras.layers.Dropout(0.5),
                             tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# 定义优化器和损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练并五折验证
model.fit(X_train, y_train, epochs=5)

# 验证集验证，evaluate返回模型的损失值和度量值
print('valid acc:\t', model.evaluate(X_valid, y_valid)[1])
print('test acc:\t', model.evaluate(X_test, y_test)[1])
