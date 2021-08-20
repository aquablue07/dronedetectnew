# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:44:27 2020

@author: vishw
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:35:49 2020

@author: vishw
"""

import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

def build_predictions(audio_dir):
    y_true =[]
    y_pred =[]
    fn_prob = {}
    print(' get features')
    for fn in tqdm(os.listdir(audio_dir)):
        rate,wav = wavfile.read(os.path.join(audio_dir, fn))
        category= fn2class[fn]
        c= classes.index(category)
        y_prob = []
        
        for i in range(0,wav.shape[0]-config.step,config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate,numcep=config.nfeat,
                     nfilt=config.nfilt,nfft=config.nfft)
            x= (x-config.min) / (config.max - config.min)
            
            if config.mode =='conv':
                x = x.reshape(1,x.shape[0],x.shape[1],1)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
            
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, fn_prob
    

df = pd.read_csv(r'C:\Users\vishw\Desktop\Book1.csv')
classes = list(np.unique(df.category)) 
fn2class = dict(zip(df.filename, df.category)) 
p_path = os.path.join('pickles', 'conv.p') 

with open(p_path,'rb') as handle:
    config =pickle.load(handle)
print(config.min,config.max)
model = load_model(config.model_path) 
y_true, y_pred , fn_prob = build_predictions(r'C:\Users\vishw\Desktop\newdataset\cleant')   
acc_score = accuracy_score(y_true=y_true, y_pred = y_pred)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.filename]
    y_probs.append(y_prob)
    for c,p in zip(classes,y_prob):
        df.at[i,c] =p

y_pred = [classes[np.argmax(y)]for y in y_probs] 
df['y_pred'] = y_pred
df.to_csv('predictions1.csv', index =False)

       