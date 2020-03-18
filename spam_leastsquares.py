# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:56:13 2019

@author: Cand5
"""

import numpy as np
import scipy.io as sio

#Code to load Matlab matrices into Python arrays
yte = sio.loadmat("yte_spam.mat")
matlabs = sio.whosmat("yte_spam.mat")
yte = yte[matlabs[0][0]]

ytr = sio.loadmat("ytr_spam.mat")
matlabs = sio.whosmat("ytr_spam.mat")
ytr = ytr[matlabs[0][0]]

xte = sio.loadmat("Xte_spam.mat")
matlabs = sio.whosmat("Xte_spam.mat")
xte = xte[matlabs[0][0]]

xtr = sio.loadmat("Xtr_spam.mat")
matlabs = sio.whosmat("Xtr_spam.mat")
xtr = xtr[matlabs[0][0]]

#Least squares classifier
xtr_t = np.transpose(xtr)
xxt = np.matmul(xtr, xtr_t)
pseudoinverse = np.matmul(xxt, xtr)
weightvector = np.matmul(pseudoinverse, np.transpose(ytr))

# Using the weightvector to calculate the class of the test set xte
test_class = np.matmul(np.transpose(weightvector), xte)

#Taking the calculated value of each mail and assigning +-1 to it:
test_classint = []
for entry in test_class[0]:
    if entry <0:
        test_classint.append(-1)
    else:
        test_classint.append(1)

#Calculating the confusion matrix
confusion_mat = np.zeros((2, 2))
ytel = yte[0]
id = 0
while id < len(test_classint):
    if test_classint[id] == 1 and test_classint[id] == ytel[id]:
        confusion_mat[0][0] += 1
    elif test_classint[id] == 1 and test_classint[id] != ytel[id]:
        confusion_mat[0][1] += 1
    elif test_classint[id] == -1 and test_classint[id] == ytel[id]:
        confusion_mat[1][1] += 1
    elif test_classint[id] == -1 and test_classint[id] != ytel[id]:
        confusion_mat[1][0] += 1
    id += 1
