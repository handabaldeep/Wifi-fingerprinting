#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:16:52 2019

@author: handabaldeep
"""
import random
import numpy as np
import sys

class kMeans:
    
    def __init__(self, tolerance=0.01, max_iterations=10):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def dist(self,a,b):
        self._sum = 0
        for i in range(len(a)):
            self._sum += (a[i] - b[i])**2
        return self._sum**0.5
    
    def centroid_dist(self,cent,p):
        min_dist = sys.maxsize
        pos = -1
        for i in range(len(cent)):
            #d = dist(p,cent[i])
            d = abs(p-cent[i])
            if d < min_dist:
                min_dist = d
                pos = i+1
        return pos
    
    def kmeans(self,input_arr,k=3):
        k_values = np.zeros(input_arr.shape[0],dtype=int)
        for i in range(len(k_values)):
            k_values[i] = random.randint(1,k)
            
        c = sorted(zip(k_values,input_arr))
        print(c)
        
        centroids = np.zeros(k)
        for j in range(self.max_iterations):
            k,i = 0,0
            while i<len(c):
                _sum,num = 0,0
                while i<len(c) and c[i][0] == k+1:
                    _sum += c[i][1]
                    num += 1
                    i += 1
                centroids[k] = _sum/num
                #print(centroids[k],_sum,num,k)
                k += 1
            
            #print(centroids)
            for i in range(len(c)):
                c[i] = (self.centroid_dist(centroids,c[i][1]),c[i][1])
            
            c.sort()
            #print(c)
        return c
            
# a = np.array([1,2,3,4,5,6,7,8,9,10])
# k1 = kMeans()
# print(k1.kmeans(a))