# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:51:48 2020

@author: vishw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cm= [[437, 42],[28, 423]]
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,alpha =0.3)
classNames = ['Negative','Positive']
plt.ylabel('Predicted label')
plt.xlabel('True label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(cm[i][j]))
plt.show()