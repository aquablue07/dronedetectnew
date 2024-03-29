#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:53:05 2018

@author: vishwa
"""
import numpy as np
import pandas as pd

class logdata:
    def __init__(self, size):
        """
        size: the dataframe size 
        """
        self.size = size
        self.df = pd.DataFrame(data = None, 
                               columns = ['Timestamp','Label','Occurance', 'Confidence'],
                              )
        self.df1 = pd.DataFrame(data = None, 
                               columns = ['Timestamp','Actuallabel','Labelpredicted_S','Labelpredicted_R','Averageprediction'],
                              )
        

    def insertdf(self, x, timestamp):
        self.x = x
        self.times = timestamp
        # default values
        self.occurance = 1
        self.confidence = 100

        self.df = self.df.append(pd.Series({
            'Timestamp': self.times, 
            'Label': self.x, 
            'Occurance': self.occurance, 
            'Confidence': self.confidence
        }), ignore_index=True)
        
        self.del_row()

        if self.df.shape[0] > 1:
            """
            Get the new Occurance & Confidence level of the current window
            Apply the change of Occurance/ Confidence values to df table 
            """
            self.occurance = self.get_occurance()
            self.confidence = self.get_confidence(self.occurance)

            print(self.df)
            print(self.occurance)
            print(self.confidence)
            self.df['Occurance'] = self.df.Label.apply(lambda x: self.occurance[x])
            self.df['Confidence'] = self.df.Label.apply(lambda x: self.confidence[x])
        
        return self.df

    
    def logdf(self, user_x, x1,x2,time):
        self.df1 = self.df1.append(pd.Series({
            'Timestamp': time,
            'Actuallabel':user_x,
            'Labelpredicted_S': x1, 
            'Labelpredicted_R': x2,
            'Averageprediction': ((x1+x2)/2) 
        }), ignore_index=True)
        

        self.df1.sort_index(inplace=True, ascending=False)
        #if int(self.df.shape[0]) > (int(i) - 1):
        #    self.df1.to_csv(file+".csv", sep='\t', encoding='utf-8')
        #iter+= 1

        return self.df1

    def savedf(self, file):
        return  self.df1.to_csv(file+".csv", sep='\t', encoding='utf-8')

    def dfempty(self):
        return self.df.empty

    def get_occurance(self):
        # group by label and count
        occ = self.df.groupby('Label').Timestamp.count().rename('Occurance').astype(int)
        return occ

    def get_confidence(self, occurance):
        """ 
        Compute confidence according to the Occurance ratio
        """
        conf = ((occurance / sum(occurance)).rename('Confidence') * 100).astype('float64')
        return conf

    def del_row(self):
        if self.df.shape[0] > int(self.size):
            self.df = self.df.tail(self.size)

    def get_result(self):
        #print("update occur df: \n")
        #print(self.df)
        """
        Output the max confidence level item, 
        if multiple, output the idxmax item: the earliest detection  
        """
        output = self.df.loc[self.df['Confidence'].idxmax()]
        #print(output)
        #return self.df.loc[self.df['Confidence'].idxmax()]
        return output
    
    def get_label_sum(self): 
        label_sum = self.df.iloc[:,1].sum(axis=0)
        return label_sum
