# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 09:53:26 2022

@author: nurul
"""
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Flatten,Dense,Dropout,LSTM
from tensorflow.keras.utils import plot_model

class EDA:
   def distplot_graph(self,con_col,df):
     for i in con_col:
        plt.figure()
        plt.plot(df[i])
        plt.plot(df['cases_new'])
        plt.legend([i,'cases_new'])
        plt.show()

class ModelEvaluation:
    def plot_hist_graph(self,hist):
            plt.figure()
            plt.plot(hist.history['mean_absolute_percentage_error'])
            plt.plot(hist.history['loss'])
            plt.legend(['Training_mean_absolute_percentage_error','Validation_mean_absolute_percentage_error'])
            plt.show()

class ModelAnalysis:
    def plot_hist_graph(self,actual_covid_cases,predicted_covid_cases):
        plt.figure()
        plt.plot(actual_covid_cases,color='blue')
        plt.plot(predicted_covid_cases,color='red')
        plt.xlabel('Act_covid_cases')
        plt.ylabel('pred_covid_cases')
        plt.legend('Actual','Predicted')
        plt.show()
        
        

      