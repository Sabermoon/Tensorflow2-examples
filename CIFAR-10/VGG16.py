"""
本代码创建 VGG16，并用于cifar-10
思考一下，效果为什么不好？
"""
import tensorflow as tf
from tensorflow.python.keras.datasets.cifar10 import load_data
from tensorflow.python.keras.applications import VGG16

vgg16 = VGG16(include_top=False, input_shape=(32,32,3))

(x_train, y_train), (x_test, y_test) = load_data()

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