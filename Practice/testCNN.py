from keras.models import load_model
model = load_model('mnistCNN.h5')

from PIL import Image
import numpy as np

img = Image.open('1.png').convert("L") #(84,84)
img = img.resize((28,28))
imgArray = np.array(img)
print(imgArray.shape) #(28,28)
imgArray = imgArray.reshape(1,28,28,1)
pred = model.predict(imgArray)
print(pred)
print(np.argmax(pred))
