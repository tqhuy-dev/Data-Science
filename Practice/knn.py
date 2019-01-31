import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('img.png').convert("L")
imgArr = np.array(img)
img =img.resize((28,28))
imgArr = np.array(img)
print(imgArr.shape)

mnist = keras.datasets.mnist

(trainImages, trainLabels), (testImages, testLabels) = mnist.load_data()

paramk = 3
x = tf.placeholder(trainImages.dtype , shape = trainImages.shape)#6000 x 28 x 28
y = tf.placeholder(testImages.dtype , shape = testImages.shape[1:])#28x28

xThresholded = tf.clip_by_value(tf.cast(x, tf.int32), 0, 1) 
yThresholded = tf.clip_by_value(tf.cast(y, tf.int32), 0, 1) 
computeL0Dist = tf.count_nonzero(xThresholded - yThresholded, axis=[1,2]) 
findKClosestTrImages = tf.contrib.framework.argsort(computeL0Dist, direction='ASCENDING') 
findLabelsKClosestTrImages = tf.gather(trainLabels, findKClosestTrImages[0:paramk]) 
findULabels, findIdex, findCounts = tf.unique_with_counts(findLabelsKClosestTrImages) 
findPredictedLabel = tf.gather(findULabels, tf.argmax(findCounts)) 

#"""
with tf.Session() as sess:
   predictedLabel = sess.run([findPredictedLabel], feed_dict={x:trainImages, y:img})
   print(predictedLabel)
#"""

"""
numErrs = 0
numTestImages = np.shape(testLabels)[0]
numTrainImages = np.shape(trainLabels)[0] # so many train images
#print(testImages[0].shape)
"""

"""
with tf.Session() as sess:
  for iTeI in range(0,numTestImages): # iterate each image in test set
    predictedLabel = sess.run([findPredictedLabel], feed_dict={x:trainImages, y:testImages[iTeI]})   
    if predictedLabel == testLabels[iTeI]:
      numErrs += 1
      print(numErrs,"/",iTeI)
      print("\t\t", predictedLabel[0], "\t\t\t\t", testLabels[iTeI])
      
      if (1):
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(testImages[iTeI])
        plt.title('Test Image has label %i' %(predictedLabel[0]))
        
        for i in range(numTrainImages):
          if trainLabels[i] == predictedLabel:
            plt.subplot(1,2,2)
            plt.imshow(trainImages[i])
            plt.title('Correctly Labeled as %i' %(testLabels[iTeI]))
            plt.draw()
            break
        plt.show()

print("# Classification Errors= ", numErrs, "% accuracy= ", 100.*(numTestImages-numErrs)/numTestImages)
"""