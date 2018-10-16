# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 23:09:59 2017

@author: konyd
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# PREPROCESSING
# Data extraction
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)


# MODEL & TRAINING
# Model initialization.
from minisom import MiniSom
som = MiniSom(10, 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# Train
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualization of the results.
from pylab import bone, pcolor, colorbar,  plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers =['o','s']
colors = ['r','g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = None,
         markersize = 10,
         markeredgewidth = 2)
show()

# Catching the fraud.
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(5,1)], mappings[(4,7)]), axis = 0)
frauds = sc.inverse_transform(frauds)