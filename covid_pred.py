# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:20:07 2022

@author: nurul
"""

#%% Import section

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential,Input
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.layers import Flatten,Dense,Dropout,LSTM

from covid_predModule import EDA
from covid_predModule import ModelEvaluation
from covid_predModule import ModelAnalysis


MMS_PATH= os.path.join(os.getcwd(),'model','mms_train.pkl')
CSV_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
CSV_TEST_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv') 
OGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))




#%% step 1) Data Loading


df = pd.read_csv(CSV_PATH)

#%% step 2) Data Inspection

df.info()
df.describe().T

# Target columns: cases_new has the presence of missing values

# To check missing values in dataset
df.isna().sum()

# only 7 columns were detect to have missing values and cases_new was not
# detect with it. Thus, we need to do missing values list in ensure the
# missing values been recognized by the dataset:
missing_values = ["?"," "]
df = pd.read_csv(CSV_PATH,na_values=missing_values)


#now, we check it again
df.isna().sum()
#now 8 columns has been inspected to have missing values compared to 7 just now
#check for the duplicate dataset
df.duplicated().sum()
# we got 0 duplicated dataset

col_names = df.columns
con_col = col_names[(df.dtypes=='int64')|(df.dtypes=='float64')]

con_col=df.drop(labels='date', axis=1)

#Visualization of dataset

eda=EDA()
eda.distplot_graph(con_col,df)

#Data insights
# from the graph, we can see that  adult cases is being impersonate by the new cases.
# This means the growth of new cases is depend by the growth pattern of adult cases, which might be 
#contribute by the factors of the increasing of unvaccinated population at the mid of 500 days of covid,increasing of
# vacc pop at the end of 500 days due to goverment order in reducing the rate percentage of citizen being affected and 
#others factors such as cluster workplace,education_centre,
#highrisk pop , import,community and religious.


#%%
# step 3) Data Cleaning

# data cleaning will be done on our main focus of dataset only,
# target columns: cases_new which contains 12 missing values. Others 7 columns will
# not have data cleaning as this is time series dataset.

# we will used interpolate method for to catter time series missing values dataset

cases_new_inter= df['cases_new'].interpolate()

#%%
# step 4) Features Selection

#scaling the dataset

#MinMaxScaler the dataset that has been interpolated

mms=MinMaxScaler()
cases_new_inter=mms.fit_transform(np.expand_dims(cases_new_inter,axis=-1))

#saving the model.pickle
with open(MMS_PATH,'wb') as file:
    pickle.dump(mms,file)

        
win_size=30
X_train=[]
y_train=[]

for i in range(win_size,np.shape(cases_new_inter)[0]):
    X_train.append(cases_new_inter[i-win_size:i,0])
    y_train.append(cases_new_inter[i,0])
    
#turn into array    
X_train=np.array(X_train)
y_train=np.array(y_train) 

#changed into 3 dim    
X_train=np.expand_dims(X_train,axis=-1)

#%% Model Development

model = Sequential()
model.add(Input(shape=(np.shape(X_train)[1],1))) # input_length # features
model.add(LSTM(32,return_sequences=(True)))
model.add(Dropout(0.3))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(1,activation='relu')) # Output layer
model.summary()

# Model arch
plot_model(model,show_shapes=(True,False),show_layer_names=True)

#%% Model compile

model.compile(optimizer='adam',loss='mse',metrics=['mean_absolute_percentage_error'])

#callbacks
L
tensorboard_callback=TensorBoard(log_dir=LOGS_PATH,histogram_freq=1)
#early_callback=EarlyStopping(monitor='val_loss',patience=5)

#%%
#Model Training
hist=model.fit(X_train,y_train,epochs=100,callbacks = [tensorboard_callback])
#saving all the loss and hist model inside

print(hist.history.keys())

#%% Model Evaluation with training dataset

me=ModelEvaluation()
me.plot_hist_graph(hist)

#%%
#Model Deployement

#data load
df_test=pd.read_csv(CSV_TEST_PATH)
#data inspection
df_test.info()

#check for missing/nan value
df_test.isna().sum()
# there's 1 missing value in cases_new                  
#check for duplicated data
df_test.duplicated().sum()
#0 duplicated dataset

# handle missing value with interpolate method
cases_new_test_inter= df_test['cases_new'].interpolate()

cases_new_test_inter.isna().sum()
#0 missing values in cases new

#change the dim to dim3
cases_new_test_inter = mms.transform(np.expand_dims(cases_new_test_inter,axis=-1))

# concatenate cases new interpolate with cases new test dataset interpolate
concat_test = np.concatenate((cases_new_inter,cases_new_test_inter),axis=0)
concat_test = concat_test[-(win_size+len(cases_new_test_inter)):]

X_test = []
for i in range(win_size,len(concat_test)):
    X_test.append(concat_test[i-win_size:i,0])

# convert X_test to array
X_test = np.array(X_test)

# Model predict using test dataset
predict_cases = model.predict(np.expand_dims(X_test,axis=-1))

#%% Plotting graph with scaling test dataset

plt.figure()
plt.plot(cases_new_test_inter,'b',label='actual_covid_cases')
plt.plot(predict_cases,'r',label='predicted_covid_cases')
plt.title('Scaled dataset')
plt.legend()
plt.show()

#%% Model Analysis with test dataset

actual_covid_cases=mms.inverse_transform(cases_new_test_inter)
predicted_covid_cases=mms.inverse_transform(predict_cases)

ma=ModelAnalysis()
ma.plot_hist_graph(actual_covid_cases,predicted_covid_cases)


print(mean_absolute_percentage_error(actual_covid_cases,predicted_covid_cases))


#Discussion:The mape that 


