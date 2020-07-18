# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:33:35 2020

@author: Kavya K
"""


import pandas as pd
import numpy as n
import untitled12 as ut                                                        #untitled12 is a file which contains related data
dataproj1=pd.read_csv("dataset.csv",low_memory=False,na_values=['nan'," "])    #using pandas library to upload/read csv files

dataproj1=dataproj1.replace(n.nan,0,regex=True)                                # Replace the strings with respective values and null columns with a 0
for i in range(1,18):
    dataproj1.replace({'Symptom_'+str(i):ut.dict},inplace=True)
dataproj1.replace({'Disease':ut.diseasedict},inplace=True)

y=dataproj1.iloc[:,0].values                                                   #to separate the columns into x ,y variable
x=dataproj1.iloc[:,1:].values

from sklearn.model_selection import train_test_split as tts                    #importing sklearn for train_test_split

trainx,testx,trainy,testy=tts(x,y,train_size=0.7,random_state=54)              #split the whole data inito train and test data

from sklearn.neighbors import KNeighborsClassifier as knn                      #the model is trained and disease is predicted based on KNN model.

model=knn()
model.fit(trainx,trainy)

predy=model.predict(testx)

print("Accuracy:",model.score(testx,testy))                                    #Accuracy is given as 99.728997% or 0.997289972899729.

from sklearn.metrics import confusion_matrix as cn
confm=cn(testy,predy)






