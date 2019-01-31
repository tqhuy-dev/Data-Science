import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(mnist_train_images , mnist_train_labels) , (mnist_test_images , mnist_test_labels) = keras.datasets.mnist.load_data()


mnist_train_images /= 255
mnist_test_images /= 255

#


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(mnist_train_images, mnist_train_labels, epochs=5)
