import os, sys, math, select
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import time as tm
import scipy.io.wavfile as wavf
import datetime
import warnings
import queue
import random

from feature_extraction import filters as fil
from feature_extraction import lpcgen as lpg
from feature_extraction import calcspectrum as csp
from feature_extraction import harmonics as hmn
from feature_extraction import fextract as fex
from feature_extraction import parsedata as par
from feature_extraction.getconfi import logdata
from feature_extraction.apicall import apicalls
from feature_extraction.specsub import reduce_noise

from sklearn import svm
from sklearn.externals import joblib
import pickle

warnings.filterwarnings("ignore")

"""set this true not to send any data to server"""
test = True

"""clf = joblib.load('input/detection_iris_new.pkl')## this is the vnear robust one"""
clf = joblib.load('input/detection_backyaardwithnoise.pkl')## this is taken at the beach
#clm = joblib.load('input/detection_new18july.pkl')
#clf1 = joblib.load('input/dronedetectionfinal_new.pkl')

rows = 10
cols = 60
winlist = []
datacount = 4

"""set this part to the number of logs you want to save before computing confidence level"""
log = logdata(datacount)

######################################################################################################
global itervalue
itervalue = 0

"""this is the script which would record data"""
def record(time = 1, fs = 44100):
    file = 'temp_out'
    duration = time
    recording = sd.rec(int(duration*fs),samplerate=fs, channels=1, blocking  = False)
    
    # this is very important to remove sound end noise... don't use sleep-_- Han
    sd.wait()

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
    data, fs = librosa.load(file+'.wav')
    os.remove(file+'.npy')
    #os.remove(file+'.wav')
    return data, fs

def dist_prediction_label(value):
    if value == 0:
        label = "far"
    elif value == 1:
        label = "midrange"
    elif value == 2:
        label = "near"
    #elif value == 3:
        #label = "vfar or nodrone"
    elif value == 3:
        label = "vnear"
    return label

noise, sfr = record()

# def drone_prediction_label(value):
#     if value == 1:
#         label = "drone"
#     elif value == 0:
#         label = "no drone"
#     return label

######################################################################################################################
"""set api and initiate calls"""
api_url = 'http://mlc67-cmp-00.egr.duke.edu/api/gardens'
apikey = None
push_url = "https://onesignal.com/api/v1/notifications"
pushkey = None
sound_url = 'http://mlc67-cmp-00.egr.duke.edu/api/soundInfos'
soundkey = None
wav_url = ' http://mlc67-cmp-00.egr.duke.edu/api/sound-clips/container/upload'
wavkey = None
LOCATION = "Drone Detector A"
send = apicalls(api_url,apikey, push_url,pushkey, sound_url, soundkey, wav_url,wavkey, LOCATION)##This initiates the push notification and mongodb database
log.insertdf(3,str(datetime.datetime.now())[:-7]) #inserted dummy value to eliminate inconsistency
i = 0
bandpass = [800,8500]#filter unwanted frequencies
prev_time= tm.time()#initiate time

waveq = queue.Queue(datacount)
recdata = np.array([],dtype="float32") # for wave concatenation
basename = "drone"
xx = [3,3,3,0,0]    # test prediction value, basic check
#xx = [0,0,1,2,3,0,0,3,3,3,0,0]    # test prediction value, advanced check

"""main code"""
try:#don't want user warnings
    while True:
        data, fs = record()
        out = reduce_noise(data,noise)
        ns = fil.bandpass_filter(data,bandpass)
        try:
            p,freq, b = hmn.psddetectionresults(data)
        except IndexError:
            pass
            b = False
        b = True
        
        if b:
            # fs = 44100#force 44100 sample rate to prediction why?
            #mfcc, chroma, mel, spect, tonnetz = fex.extract_feature(data,fs)#ns changed to raw data
            mfcc, chroma, mel, spect, tonnetz = fex.extract_feature(ns,fs)
            #a,e,k = lpg.lpc(ns,10)
            mfcc_test = par.get_parsed_mfccdata(mfcc, chroma,mel,spect,tonnetz)
            #lpc_test = par.get_parsed_lpcdata(a,k,freq)
            if test:
                x1 = random.randint(0,3)
                x1 = xx[i]
            else:
                x1 = clf.predict(mfcc_test)

            #x02 = clm.predict(mfcc_test)
            #x1 = ((x01[0]+x01[0])/2)
            #x2 = clf1.predict(lpc_test) 
            print("Drone at %s, %s"% (dist_prediction_label(int(x1)), x1))
            log.insertdf(int(x1),str(datetime.datetime.now())[:-7])
            output = log.get_result()
            print(output)
            print()            
            # collect data for datacount
            if waveq.full():    # if full, remove the first
                waveq.get()            
            waveq.put(data)     # put another data

            '''-----------uncomment if you want to save logs-----------------'''
            #log.logdf(sys.argv[1],x01[0],x02[0],str(datetime.datetime.now())[:-7])
            '''---------------------------------------------------------------'''
            if True:#i > 9:
                #print(int(output['Label']))
                #win.addstr(7,5,"Recieved a Result!")
                dt = tm.time() - prev_time
                if dt > 30 or (i>=3 and test):#send output every 30secs
                    print('sent %s'% int(output['Label']))
                    if not test:
                        send.sendtoken2(output)
                    prev_time = tm.time()
                    if int(output['Label']) == int(3) or int(output['Label']) == int(2):
            
                        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
                        sfilename = "_".join([basename, suffix])+".wav" # e.g. 'mylogfile_120508_171442'

                        if not test:
                            send.push_notify(sfilename)#when drone is detected this sends push notification to user in his app

                        print("pushed %s"% int(output['Label']))

                        while not waveq.empty():
                            recdata = np.concatenate([recdata, waveq.get()])
                        np.seterr(divide='ignore', invalid='ignore')
                        recscaled = np.int16(recdata/np.max(np.abs(recdata)) * 32767)
                        #np.save(tf.name,recscaled)
                        wavf.write(sfilename, fs, recscaled)
                        print(sfilename)

                        output.fileName = sfilename                        
                        if not test:
                            send.infosendtoken(output, sfilename)
                            send.wavsendtoken(sfilename)

                        print("file succesfully uploaded to server!")
                        #os.remove(sfilename)
                        recdata = np.array([],dtype="float32")
                        exit()
    
                    #win.addstr(8,5,"Data Sent!")
            ######################################################################################################
            # if itervalue > int(sys.argv[3]):
            #     log.savedf(sys.argv[2])
            #     exit()
            # itervalue+=1
        else:
            print("Wait for result")
        
        
        i+=1


except KeyboardInterrupt:
    pass


print('iter_num:',i)