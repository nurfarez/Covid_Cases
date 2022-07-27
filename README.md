
 <h1 align="center">

![image](![image](https://user-images.githubusercontent.com/109565405/181224746-ed1c9aaf-b3df-4630-a128-e5acbf8843a1.png))
<br>
</h1>

<h1 align="center">
  <br>
# Covid_Cases
 LSTM neural  network to predict new cases (cases_new) in Malaysia using the past 30 days of number of cases.
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
 
# Discussion
The MAPE obtained is 0.11% which more than 0.1. For the future researcher, it is recommended to reduce the layer or nodes when trained with this dataset in order to get better NN prediction model.

## Credits
Thanks to MoH-Malaysia/covid19-public: Official data on the COVID-19 for the covid dataset

