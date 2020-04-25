#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:06:26 2020

@author: mtwheeler
"""

import numpy as np
from sklearn.cluster import KMeans
from skimage.io import imread, imsave

n_colors = 2

sample_img = imread('HappyWaz.jpg')

w,h,_ = sample_img.shape

sample_img = sample_img.reshape(w*h,3)

kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(sample_img)

# find out which cluster each pixel belongs to.
labels = kmeans.predict(sample_img)

# the cluster centroids is our color palette
identified_palette = np.array(kmeans.cluster_centers_).astype(int)

# recolor the entire image
recolored_img = np.copy(sample_img)
for index in range(len(recolored_img)):
    recolored_img[index] = identified_palette[labels[index]]
    
# reshape for display
recolored_img = recolored_img.reshape(w,h,3)
imsave('HappyWaz(K=2)V2.jpg', recolored_img)