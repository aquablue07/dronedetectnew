# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:35:49 2020

@author: vishw
"""

import numpy as np
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
import pyaudio
import wave
from scipy.io import wavfile
import librosa
import time
import os
import pickle

i=0
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
    
    
df = pd.read_csv(r'C:\Users\vishw\Desktop\Book1.csv')
classes = list(np.unique(df.category)) 
fn2class = dict(zip(df.filename, df.category)) 
#p_path = os.path.join('pickles', 'conv.p') 

#with open(p_path,'rb') as handle:
#    config =pickle.load(handle)

model = load_model(r'C:\Users\vishw\Desktop\newdataset\models/conv.model') 
 
while True:
    
    FORMAT = pyaudio.paInt16
    
    CHANNELS = 2
    
    RATE = 44100
    
    CHUNK = 1024
    
    RECORD_SECONDS = 3
    
    WAVE_OUTPUT_FILENAME = "boom.wav"
    
     
    
    audio = pyaudio.PyAudio()  
    
    # start Recording
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
    
                    rate=RATE, input=True,
    
                    frames_per_buffer=CHUNK)
    
    print ("recording...")
    
    frames = [] 
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    
        data = stream.read(CHUNK)
    
        frames.append(data)
    
    print ("finished recording")
    
     
    
     
    
    # stop Recording
    
    stream.stop_stream()
    
    stream.close()
    
    audio.terminate()
    
     
    
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    
    waveFile.setnchannels(CHANNELS)
    
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    
    waveFile.setframerate(RATE)
    
    waveFile.writeframes(b''.join(frames))
    
    waveFile.close()
    y_pred,y_hat = build_predictions(r'boom.wav') 
    print(y_pred,y_hat)
  
    #resultant.wav=boom.wav
    """
    m=0, n=0
    m=m+1
    
    if y_pred ==0:
        n=n+1
        subprocess.call('sox', '-m resultant.wav boom.wav resultant.wav')
    if m==5 and n>2:
        print ('drone detected')
        m=0,n=0
    if m==5 and n<2:
        m=0,n=0   
    
    print (y_pred,y_hat)
   """     

    
           
