from allclass import *
from mytrain import *
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from random import *
img = cv2.imread("./digits.png",0)
img2 = cv2.imread("./so0.jpg",0)



# np.hsplit(row,100) cắt ảnh từ trên xuống dưới
# for row in np.vsplit(img,50) cắt nữa cái ảnh theo chiều ngang
cell = [np.hsplit(row,100) for row in np.vsplit(img,50)]



x = np.array(cell)
xx = np.array(img2)

#tạo dừ liệu train và dữ liệu test
train = x[:,:50].reshape(-1,200).astype(np.float32)
test = xx.reshape(-1,200).astype(np.float32)
# x[:,50:100] dũ liêu j của cell



# gán nhãn cho dữ liệu train

k = np.arange(10)

train_labels = np.repeat(k,250)[:,np.newaxis].astype(np.float32)


# nhận diện

knns = cv2.ml.KNearest_create()
knns.train(train,0,train_labels)
temp,ketqua,hangxom,khoangcach = knns.findNearest(test,5)
print(ketqua)
