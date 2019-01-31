import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets
def createClusteredData(N , k):
    np.random.seed(12)
    pointPerCluster = float(N)/ k
    X = []
    y = []
    for i in range(k):
        workCentroid = np.random.uniform(20000.0 , 200000.0)
        taskCentroid = np.random.uniform(20.0 , 70.0)
        print(workCentroid)
        print(taskCentroid)
        for j in range(int(pointPerCluster)):
            X.append([np.random.normal(workCentroid , 10000.0) ,
                      np.random.normal(taskCentroid , 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X , y

C = 1.0
(X , y) = createClusteredData(100,5)
svc = svm.SVC(kernel='linear' , C=C).fit(X,y)

def plotPrediction(clf):
    xx , yy =np.meshgrid(np.arange(0,250000,10),
                         np.arange(10 , 70 , 0.5))
    Z = clf.predict(np.c_[xx.ravel() , yy.ravel()])
    plt.figure(figsize = (8,6))
    Z= Z.reshape(xx.shape)
    plt.contour(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.8)
    plt.scatter(X[:,0] , X[:, 1] , c=y.astype(np.float))
    plt.show()

plotPrediction(svc)

