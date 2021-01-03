from allclass import *
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from random import *
class knn ():
    def __init__(self,traindata,ketqua,new_member):
        super().__init__()
        self.knn = cv2.ml.KNearest_create()
        self.traindata = traindata
        self.ketqua = ketqua
        self.new_member = new_member
    def train_data (self):
        self.knn.train(self.traindata,0,self.ketqua)
    def findNearest_new_member (self):
        temp,results,hangxom,khoangcach = self.knn.findNearest(self.new_member,8)
        return temp,results,hangxom,khoangcach
