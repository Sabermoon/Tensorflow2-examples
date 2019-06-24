"""
本代码创建 VGG16，并用于cifar-10
思考一下，效果为什么不好？
"""
import tensorflow as tf
import tensorflow.python.keras.layers as layers
from tensorflow.python.keras.datasets.cifar10 import load_data

# from tensorflow.python.keras.applications import VGG16
# vgg16 = VGG16(include_top=False, input_shape=(32,32,3))

# 等价实现
vgg16 = tf.keras.Sequential([
    layers.Conv2D(64, 3, padding='same', input_shape=(224, 224, 3), activation=tf.nn.relu),
    layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(strides=2, padding='same'),

    layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu),
    layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(strides=2, padding='same'),

    layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu),
    layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu),
    layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(strides=2, padding='same'),

    layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(strides=2, padding='same'),

    layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(strides=2, padding='same'),

    # layers.Flatten(),
    # layers.Dense(4096, activation=tf.nn.relu),
    # layers.Dense(4096, activation=tf.nn.relu),
    # layers.Dense(1000, activation=tf.nn.softmax)
])
# vgg16.load_weights()


(x_train, y_train), (x_test, y_test) = load_data()

# 定义 model
c_train = vgg16.predict(x_train)
c_test = vgg16.predict(x_test)

cifar = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(1, 1, 512)),
                             tf.keras.layers.Dense(1024, activation=tf.nn.relu),
                             tf.keras.layers.Dropout(0.1),
                             tf.keras.layers.Dense(256, activation=tf.nn.relu),
                             tf.keras.layers.Dropout(0.1),
                             tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
cifar.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cifar.fit(c_train, y_train, epochs=15)
cifar.evaluate(c_test, y_test)
