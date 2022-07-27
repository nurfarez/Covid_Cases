
 <h1 align="center">

![image](https://user-images.githubusercontent.com/109565405/181225008-308e35e7-f227-4ea6-8929-e969e8bc72ca.png)
<br>
</h1>

<h1 align="center">
  <br>
Covid_Cases
<br>

<h4 align="center"><a>
created by Nurul Fathihah  <br>
July 2022
</a></h4>

## About The Project
As we know, 2020 was a catastrophic year for humanity. Pneumonia of unknown aetiology was first reported in December 2019., since then, COVID-19 spread to 
the whole world and became a global pandemic. More than 200 countries were affected due to pandemic and many countries were trying to save precious lives 
of their people by imposing travel restrictions, quarantines, social distances, event postponements and lockdowns to prevent the spread of the virus. 

The scientist believed that the usage of deep learning model to predict the daily COVID cases to determine if travel bans should be imposed or rescinded. 

Thus, the main aim of this project is to develop deep learning model using LSTM neural network to predict new cases (cases_new) in Malaysia using the past 30 days
of number of cases.

## Data Insights

From the graph, we can see that  adult cases is being impersonate by the new cases.This means the growth of new cases is depend by the growth pattern of adult cases, which might be contribute by the factors of the increasing of unvaccinated population at the mid of 500 days of covid,increasing of vacc pop at the end of 500 days due to goverment order in reducing the percentage rate of citizen being infected and others factors such as cluster workplace,education_centre,highrisk pop ,import,community and religious. 
 
![adult vs new](https://user-images.githubusercontent.com/109565405/181197655-bb1f4a8d-f7ea-4684-9076-e3081089036b.png)

![unvax vs new](https://user-images.githubusercontent.com/109565405/181198216-915b7491-8de0-438b-abe9-30e294b42e38.png)
 
![pvax vs new](https://user-images.githubusercontent.com/109565405/181198318-15283fd8-a9dd-402c-8f33-6649852bd246.png)

![fvax vs new](https://user-images.githubusercontent.com/109565405/181198627-45ff23ed-12d4-485b-a1cc-e0ff000b7523.png)

![workplace vs new](https://user-images.githubusercontent.com/109565405/181198719-db079120-84e4-4be2-8002-6ea83cee5b0d.png)

![religious vs new](https://user-images.githubusercontent.com/109565405/181198766-48469994-5242-4604-89db-84f0f18d26d5.png)

 
![import vs new](https://user-images.githubusercontent.com/109565405/181198817-8f2b0932-c9ef-4ad1-af8d-771718536219.png)


![detention vs new](https://user-images.githubusercontent.com/109565405/181198880-8172fb7c-3ffc-490f-a9b5-fc19b5249489.png)
 

![highrisk vs new](https://user-images.githubusercontent.com/109565405/181198850-26c67ffa-2df9-4e73-bc73-b707ece0c4bd.png)

 
## Libraries that been used

- matplotlib.pyplot 
- numpy
- pandas
- os
- sklearn
- datetime
- tensorflow
- sklearn
- seaborn

## Data cleaning
Our main target of this model is: cases_new.There are missing values found in train dataset for this features: 12 missing values and test dataset: 1 missing value.
Since this is time series dataset, the best way to catter with missing values is by interpolation method. 

## Features Selection
Since this is time series dataset, there no features need to be selected where our main focus is to predict cases_new

## Data Preprocessing

1. We are MinMaxScaling the interpolate cases_new for both dataset train and test by fit and transform it the interpolate cases_new of train dataset and transform the interpolate cases_new of test dataset
2.We set the window size as 30 as per 30 days requested for the analysis

## Model Development
 We build the model by sequential approach with 2 layers, 32 nodes and dropout rate of 0.2 with epoch of 500.
 
 The Model arch was plotted:
![model](https://user-images.githubusercontent.com/109565405/181232244-a7b2b37c-bfda-486d-9b11-210b2367f948.png)


the running model:


![running model](https://user-images.githubusercontent.com/109565405/181232624-991c69e8-ac0a-4e61-a49e-ef7ab4273d2b.PNG)


The epochs of mape and epochs loss were plotted in tensorboard:

![tensorboard](https://user-images.githubusercontent.com/109565405/181232748-2c6dbb25-d378-455e-a08d-3ffb5aa8fa76.PNG)


The mape was plotted using training dataset:

![training vs validation mape train dataset](https://user-images.githubusercontent.com/109565405/181233087-1aa7cae3-460a-48da-ac57-cb3eb4a6da1c.png)

 
 # Discussion

 This model is plotted with inversed test dataset also:
  ![act_covid vs pred_covid](https://user-images.githubusercontent.com/109565405/181235485-338f4b0e-2345-4241-ae86-9baa24346bff.png)

 
![mape](https://user-images.githubusercontent.com/109565405/181234738-4093321d-04c3-4d5c-9a5c-35a84ef9d6dc.PNG)

The MAPE obtained is 0.11% which more than 0.1%. This means the model is still a good model and be able to predict the new cases of covid19.However, for the future researcher, it is recommended to reduce the layer or nodes when trained with this dataset in order to get better NN prediction DL model .

## Credits
Thanks to MoH-Malaysia/covid19-public: Official data on the COVID-19 for the covid dataset

