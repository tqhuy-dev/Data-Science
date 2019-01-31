import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

trainImages = []
trainLabels = ['truck' , 'car','car','car','car','truck','truck']
for x in range(7):
    img = Image.open(str(x) + 'c.jpg').convert("L")
    img=img.resize((400,400))
    imgArr = np.array(img)
    trainImages.append(imgArr)

trainImagesArr = np.array(trainImages)
print(trainImagesArr.shape)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(trainImagesArr.shape[1], trainImagesArr.shape[2], 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(trainImagesArr, trainLabels, validation_data=(testImages, testLabels), epochs=10, batch_size=200)
model.save('mnistCNN.h5')