import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa


def envelope(y, rate, threshold):     
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window = int(rate/10), min_periods = 1,center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask       
def calc_fft(y, rate):
    n=len(y)
    freq = np.fft.rfftfreq(n, d=1/rate) 
    Y = abs(np.fft.rfft(y)/n)
    return (Y,freq)       

df = pd.read_csv(r'C:\Users\vishw\Desktop\Book2.csv')
df.set_index('filename',inplace=True)

for f in df.index:
    rate, signal =wavfile.read(r'C:\Users\vishw\Desktop\soundsfinal/'+f)
    df.at[f,'length'] = signal.shape[0]/rate
    
classes = list(np.unique(df.category))
class_dist = df.groupby(['category']) ['length'].mean()   
fig, ax =plt.subplots()
ax.set_title('Class Distribution', y =1.08)
ax.pie(class_dist,labels=class_dist.index,autopct='%1.1f%%',
       shadow=False, startangle=90)
ax.axis('equal')
plt.show()
df.reset_index(inplace=True) 

signals = {}
fft = {} 
fbank = {}
mfccs = {}
  
for c in classes:
    wav_file = df[df.category == c].iloc[0,0]
    signal, rate = librosa.load(r'C:\Users\vishw\Desktop\soundsfinal/'+wav_file, sr=44100)
    mask = envelope(signal,rate, 0.0005)
    signal = signal[mask]
    signals[c] =signal
    fft[c] =calc_fft(signal,rate)
    
    bank = logfbank(signal[:rate], rate, nfilt=64,nfft=1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate],rate,numcep=64, nfilt=64,nfft=1103).T
    mfccs[c] =mel
    
if len(os.listdir('cleanqualtrics'))==0:
    for f in tqdm(df.filename):
        signal, rate = librosa.load(r'C:\Users\vishw\Desktop\soundsfinal/'+f, sr =44100)
        mask = envelope(signal, rate, 0.0005)
        wavfile.write(filename='cleanqualtrics/'+f,rate=rate, data=signal[mask])
