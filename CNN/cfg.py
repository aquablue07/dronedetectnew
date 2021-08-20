# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:35:04 2019

@author: vishw
"""
import os

class config:
    def __init__(self,mode='conv', nfilt=64, nfeat=64, nfft=1200, rate=44100):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.rate = rate
        self.nfft = nfft
        self.step = int(rate/10) 
        self.model_path = os.path.join('models', mode + '.model')
        self.p_path = os.path.join('pickles', mode + '.p')
        
