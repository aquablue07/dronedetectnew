# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 04:17:48 2020

@author: vishw
"""

import subprocess
import numpy as np
import sys
import time as tm
import datetime
from apirenew import apicalls
import os


from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
import librosa
import sox


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

def dist_prediction_label(value):
    if value == 0:
        label = "far"
    elif value == 1:
        label = "midrange"
    elif value == 2:
        label = "near"
    elif value == 3:
        label = "vfar or nodrone"
    elif value == 4:
        label = "vnear"
    return label
basename = "drone"
while(True):
    tm.sleep(5)
######################################################################################################################    """set api and initiate calls"""
    # api_url = 'http://mlc67-cmp-00.egr.duke.edu/api/events' ##This is dukes server which Chunge created
    api_url = 'http://mlc67-cmp-00.egr.duke.edu/api/gardens' ##This is garden server  

    apikey = None
    push_url = "https://onesignal.com/api/v1/notifications"
    pushkey = None
    sound_url = 'http://mlc67-cmp-00.egr.duke.edu/api/soundInfos'
    soundkey = None
    wav_url = ' http://mlc67-cmp-00.egr.duke.edu/api/sound-clips/gardens/upload'
    wavkey = None
    LOCATION = "Drone Detector A"
    send = apicalls(api_url,apikey, push_url,pushkey, sound_url, soundkey, wav_url,wavkey, LOCATION)##This initiates the push notification and mongodb database
    #df = pd.read_csv(r'Book1.csv')
    #classes = list(np.unique(df.category)) 
    #fn2class = dict(zip(df.filename, df.category)) 
    
    #model = load_model(r'conv.model')
    #rf
    try:
        subprocess.call(['hackrf_sweep', '-f 2400:2500', '-l 40', '-g 20', '-r newbie.csv'],
                              timeout=5)
    except subprocess.TimeoutExpired: 
        print('subprocess has been killed on timeout')
    else:
        print('subprocess has exited before timeout')
                
        
    #from static_global_max import get_global_max

    def fifty_shots(startline, file, max_values):
        result = {}
        states = []

        # step 1: read 20 * 50 lines and save in 20 keys dictionary
        for row in file.readlines()[startline: startline + 1000]:
            row = row.split(",")
            key = int(int(row[2]) / 1000000)
            if key not in result:
                result[key] = []
            temp = [float(i) for i in row[6:]] #append values
            temp.append(row[1]) #append cur timepoint
            result[key].append(temp)

        # step 2: check each key (20 frequencies)
        for key in max_values:
            temp = np.array(result[key])
            cur_state = [key]

            # step 3: check each columns
            for i in range(5):
                # check if this key (frequency) ever exist from 5 gains
                ith_column_state = check(temp, i, max_values[key])
                cur_state.append(ith_column_state)

            # step 4: check current state
            for entry in cur_state[1:]:
                if entry[0] == "Exist":
                    states.append([cur_state[0], entry[1]])
                    break

        # step 5: return valid states of each frequency in this 50 shots
        #print(states)
        return states

    def check(column, index, max_value, threshold = 20):
        """
        cannot use "count" as function name which may cause:
        TypeError: 'int' object is not callable
        """
        count = 0
        timepoint = "Not Exist"

        # step 1: check each entry of current column (size == 50)
        """
        have to transfer k to float because 
        it is str type(because the last column is string type)
        """
        for k in column:
            if float(k[index]) > max_value:
                count += 1
                timepoint = k[-1] if count == 1 else timepoint
        if count >= threshold:
            return ["Exist", timepoint]
        return ["Not Exist"]

    def checkhelper(states, threshold = 5):
        """
        check if the frequency is continuous.
        :param states: the valid states of 50 shots
        :return: if we have continuous 20mHz band size frequency
        """
        temp = np.zeros(20)
        for state in states:
            pos = int((state[0] - 2400) / 5)
            temp[pos] += 1
        maxlen = 0
        curlen = 0
        for count in temp:
            if count >= 1:
                curlen += 1
                maxlen = max(maxlen, curlen)
            else:
                curlen = 0
        #print(temp)
        return maxlen >= threshold

    if __name__ == "__main__":
        # step 1: get static global max value of each gain column
        #max_values = get_global_max()
        #max_values = get_global_max('withoutdrone.csv')
        max_values = {2400: -45, 2410: -45, 2405: -45, 2415: -45, 2420: -45, 2430: -45, 2425: -45, 2435: -45,
         2440: -45, 2450: -45, 2445: -45, 2455: -45, 2460: -45, 2470: -45, 2465: -45, 2475: -45,
         2480: -45, 2490: -45, 2485: -45, 2495: -45}
        # step 2: read lines from the file every 50 times
        start_line = 0
        count = 0
        every_two_seconds = []
        if True:
            file = open(' newbie.csv', 'r')
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M")
                
            subprocess.call(['arecord -r 44100 -c 1 -d 10 test.wav'],shell =True)
            os.popen('cp test.wav resultant.wav')
            tm.sleep(1)
            os.rename(r'test.wav',r'drone'+str(suffix)+'.wav')
            #file = open('withdrone24gcorrect.csv', 'r')
            # step 3: exclude the status of "file end"
            try:
                # step 4: check if there exists drone operation
                states = fifty_shots(start_line, file, max_values)
                #print(len(states))
                # step 5: update start_line index
                start_line += 1000

                # step 6: check band size
                #print(states)
                if checkhelper(states):
                    print(states[0][1], "has drone")
                    try:#don't want useless user warnings
                        fileName = r'drone'+str(suffix)+'.wav'
                        output = { 
                                'Timestamp': str(datetime.datetime.now())[:-7], 
                                'Label': "2", # near
                                'Occurance': "", 
                                'Confidence': 66,
                                'fileName' : fileName }

                        print('sent %s'% int(output['Label']))
                        send.sendtoken(output)#This line sends the log to srver(recent detection with confidence)
                        if int(output['Label']) == int(4) or int(output['Label']) == int(2):
                            send.push_notify(fileName)#when drone is detected this sends push notification to user in his app
                            print("pushed %s"% int(output['Label']))
                            send.wavsendtoken(fileName)
                                   
                    except KeyboardInterrupt:
                        pass
                
                else:
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
                    print (n)
                    if n<=2:
                        
                        fileName = r'drone'+str(suffix)+'.wav'
                        output = { 
                                'Timestamp': str(datetime.datetime.now())[:-7], 
                                'Label': "2", # near
                                'Occurance': "", 
                                'Confidence': 66,
                                'fileName' : fileName }

                        print('sent %s'% int(output['Label']))
                        send.sendtoken(output)#This line sends the log to srver(recent detection with confidence)
                        if int(output['Label']) == int(4) or int(output['Label']) == int(2):
                            send.push_notify(fileName)#when drone is detected this sends push notification to user in his app
                            print("pushed %s"% int(output['Label']))
                            send.wavsendtoken(fileName)
                        print ('drone detected acoustic')
            
                os.remove(r'drone'+str(suffix)+'.wav')
            except BaseException:
                sys.stderr.write("Out of file bound!")
                sys.exit(0)


