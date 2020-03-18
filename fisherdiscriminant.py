# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:56:13 2019

@author: pebo2
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import svm_util as su

#Loading the data into Python arrays
moons = sio.loadmat("moons_dataset.mat")
matlabs = sio.whosmat("moons_dataset.mat")
moonvectors = moons[matlabs[0][0]]
moonclass = moons[matlabs[1][0]]

#Splitting the set into training and test set, with a ration of 3:1
training_set = moonvectors[0:300][:]
test_set = moonvectors[300:400][:] 
training_class = moonclass[0][0:300]

def pointmean(x, var, kons, cl_set):
    """Takes the vector x and the constants var and kons, then calculates
    the sum of the gaussian functions for every training vector and divides
    by kons to get the average. It does this for both spam and mail classes
    and returns the probability of x being in either.
    kons: 1/(N*2pi^(L/2)*h**2)
    var: 1/(2h^2)
    x: a L-dimensional vector
    cl_set: the set of data points belonging to class i
    """
    var = 2*var**2
    expsum = 0
    for vec in cl_set:
        dist = x - vec
        xxt = np.matmul(np.transpose(dist), dist)
        prob = np.exp(-xxt/var)
        expsum += prob
    return expsum/(kons*len(cl_set))

moon_red = []
moon_blue = []
id = 0

while id < len(training_class):
    pst = training_set[id,:]
    if training_class[id] == 1:
        moon_red.append(pst)
    else:
        moon_blue.append(pst)
    id += 1

#Calculating the means of each class
moon_red = np.asarray(moon_red)
moon_blue = np.asarray(moon_blue)
blue_mean = np.mean(moon_blue, axis=0)
blue_var = np.var(moon_blue, axis = 0)
red_mean = np.mean(moon_red, axis=0)
red_var = np.var(moon_red, axis = 0)
mean_vector = blue_mean - red_mean

gauss_blue = su.gaussian_kernel(moon_blue, None, 0.5)


#Calculating the M matrix
kons = 1/(2*np.pi*0.5**2)
mean_b = []
mean_r = [] 
for x in training_class:
    res = pointmean(x, 0.1, kons, moon_red)
    mean_r.append(res)
for x in training_class:
    res = pointmean(x, 0.1, kons, moon_blue)
    mean_b.append(res)

mean_b = np.asarray(mean_b)
mean_r = np.asarray(mean_r)
mb_mr = mean_b - mean_r

matM = np.outer(mb_mr, mb_mr)

#calculating the N matrix:
matI1b = np.eye(len(moon_blue)) - np.ones((len(moon_blue) ,len(moon_blue) ))/len(moon_blue) 
matI1r = np.eye(len(moon_red)) -np.ones((len(moon_red) ,len(moon_red) ))/len(moon_red)  
  
matKb = np.zeros((300, len(moon_blue)))
    
#Solving the eigenvalue problem for S_B/S_W
#sq_inv = np.linalg.inv(matN)
#swsb = np.matmul(sq_inv, matM)
#eig_values, eig_vectors = np.linalg.eigh(swsb)
#eig_vect = eig_vectors[1][:]-eig_vectors[0][:]

#Preparing data for plotting
xrkoor, yrkoor = np.split(moon_red, 2, 1)
xbkoor, ybkoor = np.split(moon_blue, 2, 1)


koor = plt.figure()
graf = koor.add_subplot(111)
plt.arrow(*red_mean, *mean_vector, color='purple')
#plt.arrow(*red_mean, *eig_vectors[1][:], color='green')
#plt.arrow(*red_mean, *eig_vectors[0][:], color='green')
graf.plot(xrkoor, yrkoor, 'ro', color='salmon')
graf.plot(xbkoor, ybkoor, 'ro', color='aqua')
#graf.plot(x_vals, y_vals, "k:", label="0-distr.", color='g')
