# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
pageSeeds = np.random.normal(3.0,1.0,100)
purchaseAmount = np.random.normal(50.0,30.0,100)/pageSeeds

trainDataX = pageSeeds[:80]
testDataX = pageSeeds[80:]

trainDataY = purchaseAmount[:80]
testDataY = purchaseAmount[80:]

testx = np.array(testDataX)
testy = np.array(testDataY)
p4 = np.poly1d(np.polyfit(testx , testy, 8))

xp = np.linspace( 0 , 7 , 100)
axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0,200])

plt.scatter(testx , testy)
plt.plot(xp, p4(xp) , c='r')
plt.show()