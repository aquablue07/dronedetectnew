# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 04:50:57 2019

@author: vishw
"""


import os, sys, math, select
import librosa
import sounddevice as sd
import time as tm
import scipy.io.wavfile as wavf
import datetime
from python_speech_features import mfcc
import warnings
from keras.models import model_from_yaml
from cfg import config
from keras.models import load_model
import pickle
import os
import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
from sklearn.metrics import accuracy_score

def build_predictions(audio):
    y_true=[]
    y_pred=[]
    fn_prob ={}
    x= []
    _min, _max = float('inf'), float('-inf')
    print(' extracting features from audio')
    rate,wav = wavfile.read(audio)
    
    y_prob = []
    
    x_mfcc= mfcc(wav,rate,numcep =64,
        nfilt=64,nfft=1103).T
    _min = min(np.amin(x_mfcc),_min)
    _max = max(np.amax(x_mfcc),_max)
    x.append(x_mfcc)
    x=np.array(x)    
    x = (x-_min)/(_max-_min) 
    x =x.reshape(x.shape[0],x.shape[1],x.shape[2],1)  
    y_hat = model.predict(x)
    y_prob.append(y_hat)
    #y_true.append(c)
#       fn_prob[fn] = np.mean(y_prob)   
   # print (y_true)
    #print (y_hat)
    return y_prob   
            

model = load_model(r'C:\Users\vishw\Desktop\newdataset\models\conv.model')


while True:


    file = 'temp_out'
    
    duration = time =1
    fs=44100
    
    recording = sd.rec(int(duration*fs),samplerate=fs, channels=1, blocking  = False)
    
    for i in range(time):
    
        i += 1
    
        tm.sleep(1)
    
    recording = recording[:,0]
    
    np.save(file,recording)
    
    np.seterr(divide='ignore', invalid='ignore')
    
    scaled = np.int16(recording/np.max(np.abs(recording)) * 32767)
    
    wavf.write(file+'.wav', fs, scaled)
    
    """uncomment the following if you want to save data"""
    
    ##################################################################################################
    
    #wavf.write(file+'_'+str(itervalue)+'.wav', fs, scaled)
    
    #data, fs = librosa.load(file+'_'+str(itervalue)+'.wav')
    
    ######################################################################################################
    
    #data, fs = librosa.load(file+'.wav')
    print (build_predictions(file+'.wav'))
    
    os.remove(file+'.npy')
    
    os.remove(file+'.wav')
    
                 
    
    
    
    
    

