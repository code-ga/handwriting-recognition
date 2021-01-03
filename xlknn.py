from allclass import *
from mytrain import *
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from random import *
traindata = np.random.randint(0,100 , (25,2)).astype(np.float32)
ketqua = np.random.randint(0,2, (25,1)).astype(np.float32)
red = traindata[ketqua.ravel()==1]
blue = traindata[ketqua.ravel()==0]
new_member = np.random.randint(0,100, (1,2)).astype(np.float32)
plt.scatter(red[:,0],red[:,1],100,"red","^")
plt.scatter(blue[:,0],blue[:,1],100,"blue","s")
plt.scatter(new_member[:,0],new_member[:,1],100,"green","o")
knn = knn(traindata,ketqua,new_member)
knn.train_data()
temp,results,hangxom,khoangcach = knn.findNearest_new_member()
print(results)
plt.show()
