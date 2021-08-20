# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:35:49 2020

@author: vishw
"""

import numpy as np
import os
import warnings
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
import wave
from scipy.io import wavfile
import librosa
import subprocess
import sox
import time
import os


#remove stray noises

def envelope(y,rate,threshold):
    mask=[] 
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=4410,min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def build_predictions(audio):
    y_pred =[]
    
    wav,rate = librosa.load(audio, sr= 44100)
    mask =envelope(wav,rate,0.0005)
    wav=wav[mask]
    y_prob = []
    
    for i in range(0,wav.shape[0]-4410,4410):
        sample = wav[i:i+int(4410)]
        x = mfcc(sample, rate,numcep=64,
                 nfilt=64,nfft=1200)
        x= (x+221.83284498388426) / (209.99365130211018+221.83284498388426)                
        x = x.reshape(1,x.shape[0],x.shape[1],1)
        y_hat = model.predict(x)
        y_prob.append(y_hat)
        y_pred.append(np.argmax(y_hat))
        return y_pred,y_hat
    
    
df = pd.read_csv(r'Book1.csv')
classes = list(np.unique(df.category)) 
fn2class = dict(zip(df.filename, df.category)) 


model = load_model(r'conv.model')
subprocess.call(['arecord', '-r 44100', '-c 1', '-d 10', 'resultant.wav'])


print('start')
subprocess.run(['sox resultant.wav boom1.wav trim 0 2'],shell=True)
y_pred,y_hat = build_predictions(r'boom1.wav')
i=y_pred[0]
subprocess.run(['sox resultant.wav boom2.wav trim 2 4'],shell=True)
y_pred,y_hat = build_predictions(r'boom2.wav')
j=y_pred[0]
subprocess.run(['sox resultant.wav boom3.wav trim 4 6'],shell=True)
y_pred,y_hat = build_predictions(r'boom3.wav')
k=y_pred[0]
subprocess.run(['sox resultant.wav boom4.wav trim 6 8'],shell=True)
y_pred,y_hat = build_predictions(r'boom4.wav')
l=y_pred[0]   
subprocess.run(['sox resultant.wav boom5.wav trim 8 10'],shell=True)
y_pred,y_hat = build_predictions(r'boom5.wav')
m=y_pred[0]
n= i+j+k+l+m
print(n)
if n<=2:
    print ('drone detected acoustic')
    print (n)
    



    
  
           
    
    

    
           
