# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:56:13 2019

@author: pebo2
"""

import numpy as np
import svm_util as su
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

gauss_tr, gauss_te = su.gaussian_kernel(np.transpose(xtr), np.transpose(xte), 0.5)

alphas, bias = su.smo_simplified(gauss_tr, ytr, 1.7)

test_class = np.matmul((alphas*ytr), np.transpose(gauss_te))
test_class = np.add(test_class, bias) 

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
