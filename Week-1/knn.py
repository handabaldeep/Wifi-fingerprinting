#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:15:22 2019

@author: handabaldeep
"""

import numpy as np

def knn_classification(input_arr, output_arr, test_arr, k=1):
    
    dist_list = np.zeros(input_arr.shape[0])
    vals_list = np.zeros(input_arr.shape[0])
    for i in range(input_arr.shape[0]):
        dist = 0
        for j in range(input_arr.shape[1]):
            dist += (input_arr[i][j]-test_arr[j])**2
        dist_list[i] = dist**(0.5)
        vals_list[i] = output_arr[i]
    
    #print(dist_list,vals_list)
    values = sorted(zip(dist_list,vals_list))
    #print(values)
    v = []
    for i in range(k):
        v.append(values[i][1])
    v.sort()
    #print(v)

    repeat, max_iter = 1,1
    ret = v[0]
    for j in range(len(v)-1):
        if j == k:
            break
        if v[j] == v[j+1]:
            repeat += 1
        else:
            repeat = 1
        if repeat >= max_iter:
            max_iter = repeat
            ret = v[j]
        #print(repeat,max_iter)
    
    return ret

def knn_regression(input_arr, output_arr, test_arr, k=1):
     
    dist_list = np.zeros(input_arr.shape[0])
    vals_list = np.zeros(input_arr.shape[0])
    for i in range(input_arr.shape[0]):
        dist = 0
        for j in range(input_arr.shape[1]):
            dist += (input_arr[i][j]-test_arr[j])**2
        dist_list[i] = dist**(0.5)
        vals_list[i] = output_arr[i]
    
    #print(sorted(zip(dist_list,vals_list)))
    values = sorted(zip(dist_list,vals_list))
    #print(values)
    v = []
    for i in range(k):
        v.append(values[i][1])
    #print(v)
    
    return sum(v)/len(v)

input_a = np.array([[0,3],[2,2],[3,3],[-1,1],[-1,-1],[0,1]])
output_a = np.array([1,1,1,-1,-1,-1])
test_a = np.array([1,2])
print('For k=1:',knn_classification(input_a,output_a,test_a))
print('For k=3:',knn_classification(input_a,output_a,test_a,3))
print()

input_b = np.array([[1,2,3,0],[1,4,2,3],[-2,3,-4,3],[-1,1,-3,2]])
output_b = np.array([1,1,-1,-1])
test_b = np.array([0,1,0,1])
print('For k=1:',knn_classification(input_b,output_b,test_b))
print('For k=3:',knn_classification(input_b,output_b,test_b,3))
print()

input_c = np.array([[1,2,3,0],[1,4,2,3],[-2,3,-4,3],[-1,1,-3,2]])
output_c = np.array([-2,2,0,1])
test_c = np.array([0,1,0,1])
print('For k=1:',knn_regression(input_c,output_c,test_c))
print('For k=3:',knn_regression(input_c,output_c,test_c,3))