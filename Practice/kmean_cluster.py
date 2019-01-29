# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:55:35 2019

@author: DELL
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

def createClusteredData(N , k):
    np.random.seed(12)
    pointPerCluster = float(N)/ k
    X = []
    for x in range(k):
        workCentroid = np.random.uniform(80.0 , 400.0)
        taskCentroid = np.random.uniform(10.0 , 30.0)
        print(workCentroid)
        print(taskCentroid)
        for j in range(int(pointPerCluster)):
            X.append([np.random.normal(workCentroid , 200.0) ,
                      np.random.normal(taskCentroid , 2.0)])
    X = np.array(X)
    return X

data = createClusteredData(900 , 4)
model = KMeans(n_clusters=4)
model = model.fit(scale(data))
print(len(model.labels_))
data00 = []
data01 = []
data02 = []
data03 = []
for x in range(900):
    if(model.labels_[x] == 0):
        data00.append(data[x])
    if(model.labels_[x] == 1):
        data01.append(data[x])
    if(model.labels_[x] == 2):
        data02.append(data[x])
    if(model.labels_[x] == 3):
        data03.append(data[x])
        
data00 = np.array(data00)
data01 = np.array(data01)
data02 = np.array(data02)
data03 = np.array(data03)


plt.figure(figsize = (8,6))
plt.plot(data00[: , 0] , data00[: , 1] ,'b^' , markersize = 4, alpha = .8)
plt.plot(data01[: , 0] , data01[: , 1] ,'go' , markersize = 4, alpha = .8)
plt.plot(data02[: , 0] , data02[: , 1] ,'rs' , markersize = 4, alpha = .8)
plt.plot(data03[: , 0] , data03[: , 1] ,'yo' , markersize = 4, alpha = .8)

plt.plot()
#plt.scatter(data[: , 0] , data[: , 1] , c= model.labels_.astype(np.float))
plt.show()

