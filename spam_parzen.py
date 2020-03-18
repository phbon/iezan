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

#Code to split the training set into separate arrays
xtrsplit = np.hsplit(xtr, 2300)

#Splitting the 57-dim vectors into spam and regular mail:
xtr_spam = []
xtr_mail = []
id = 0
while id < len(ytr[0]):
    pst = xtrsplit[id]
    if ytr[0][id] == 1:
        xtr_mail.append(pst)
    else:
        xtr_spam.append(pst)
    id += 1


def pointmean(x, var, kons):
    """Takes the vector x and the constants var and kons, then calculates
    the sum of the gaussian functions for every training vector and divides
    by kons to get the average. It does this for both spam and mail classes
    and returns the probability of x being in either.
    kons: 1/(N*2pi^(L/2)*h^L)
    var: 1/(2h^2)
    x: a L-dimensional vector
    """
    var = 2*var**2
    p_spam = 0
    for vec in xtr_spam:
        dist = x - vec
        xxt = np.matmul(np.transpose(dist), dist)
        prob = np.exp(-xxt/var)
        p_spam += prob
        
    p_mail = 0
    for vec in xtr_mail:
        dist = x - vec
        xxt = np.matmul(np.transpose(dist), dist)
        prob = np.exp(-xxt/var)
        p_mail += prob  
    
    return p_spam/(kons*len(xtr_spam)), p_mail/(kons*len(xtr_mail))

#Defining the constant for the Gauss functions
variance = 0.8
dimensions = 57
konstant = 1/((2*np.pi)**(dimensions/2)*variance**dimensions)
spam_prop = len(xtr_mail)/len(xtr_spam)

#Splitting the test set into 2301 vectors:
xtesplit = np.hsplit(xte, 2301)

# Array to hold the classification results for the vectors in xte
test_class = []

# Loop to go through the test vectors and assign to spam or mail class
for vector in xtesplit:
    p_spam, p_mail = pointmean(vector, variance, konstant)
# Assign to class 1 if P_1/P_2 > 1
    if p_spam > p_mail*spam_prop:
        test_class.append(-1)
    else:
        test_class.append(1)

#Calculating the confusion matrix
confusion_mat = np.zeros((2, 2))
ytel = yte[0]
id = 0
while id < len(test_class):
    if test_class[id] == 1 and test_class[id] == ytel[id]:
        confusion_mat[0][0] += 1
    elif test_class[id] == 1 and test_class[id] != ytel[id]:
        confusion_mat[0][1] += 1
    elif test_class[id] == -1 and test_class[id] == ytel[id]:
        confusion_mat[1][1] += 1
    elif test_class[id] == -1 and test_class[id] != ytel[id]:
        confusion_mat[1][0] += 1
    id += 1





















