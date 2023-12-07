import pandas as pd
import numpy as np
import matplotlib as mtp
import seaborn as snb
import datetime

# 1)Load the data and check the details about the data
data_set = pd.read_csv('Used_car_price.csv')
# show first 5 entries (head) and last 5 entries
print(data_set.head())
print(data_set.tail())
# information about the data, where we can execute no of columns and rows,data type and memory usage
print(data_set.info())
print(data_set.shape)
# check the null values contain in the data
print(data_set.isnull().sum())
# Statistical function
print(data_set.describe())
# 2)preprocessing the data
# change the columns names
data_set.rename(columns={'Company Name':'Company_Name','Engine Type':'Engine_Type','Engine Capacity':'Engine_Capacity','Transmission Type':'Transmission_Type','Registration Status':'Registration_Status'},inplace=True)
# column Model Year is converted into age of the used car
date_time = datetime.datetime.now()
data_set['Age'] = date_time.year-data_set['Model Year']
# drop the unwanted columns
data_set.drop(['Model Year','Unnamed: 0','Model Name','Location','Body Type','Color'], axis=1, inplace=True)
# Remove the outliers
# snb.boxplot(data_set['Price'])
sorted(data_set['Price'],reverse=True)
print(data_set.head())
# remove the outlier
data_set = data_set[~(data_set['Price'] >= 77500000)]
print(data_set.shape)
# Encoding the categorical columns
print(data_set['Engine_Type'].unique())
data_set['Engine_Type'] = data_set['Engine_Type'].map({'Petrol': 0, 'Diesel': 1,'Hybrid': 2})
print(data_set['Assembly'].unique())
data_set['Assembly'] = data_set['Assembly'].map({'Imported': 0, 'Local': 1})
print(data_set['Transmission_Type'].unique())
data_set['Transmission_Type'] = data_set['Transmission_Type'].map({'Automatic': 0, 'Manual': 1})
print(data_set['Registration_Status'].unique())
data_set['Registration_Status'] = data_set['Registration_Status'].map({'Un-Registered': 0, 'Registered': 1})
# independent and dependent variables
X = data_set.drop(['Company_Name','Price'],axis=1)
y = data_set['Price']
# Splitting the dataset into training and testing set
# In the data dependent variable price is a continuous values , so regression models are use
# import regression models

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state =42)
from xgboost import XGBRegressor
xgb= XGBRegressor()
xgb.fit(X_train,y_train)
y_predict = xgb.predict(X_test)
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
from sklearn.metrics import r2_score
# predicting the accuracy score
score=r2_score(y_test,y_predict)
print("r2 socre is ",score*100,"%")