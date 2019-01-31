import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

(trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()
trainImages = trainImages.reshape(trainImages.shape[0],trainImages.shape[1] ,
                                  trainImages.shape[2] ,1).astype('float32')
testImages = testImages.reshape(testImages.shape[0],testImages.shape[1] ,
                                  testImages.shape[2] ,1).astype('float32')

print(trainLabels.shape)
trainImages /= 255
testImages /=255

numberOfClass = 10
trainLabels = np_utils.to_categorical(trainLabels, numberOfClass)
testLabels = np_utils.to_categorical(testLabels, numberOfClass)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(trainImages.shape[1], trainImages.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(numberOfClass, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(trainImages, trainLabels, validation_data=(testImages, testLabels), epochs=10, batch_size=200)
model.save('mnistCNN.h5')

metrics = model.evaluate(testImages, testLabels, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)