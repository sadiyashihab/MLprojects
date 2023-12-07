import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import numpy as np

data_set= pd.read_csv('Used_car_price.csv')
print(data_set.head())
print(data_set.tail())

print(data_set.info())
print(data_set.shape)
print(data_set.isnull().sum())
print(data_set.describe())
data_set.rename(columns={'Company Name':'Company_Name','Engine Type':'Engine_Type','Engine Capacity':'Engine_Capacity','Transmission Type':'Transmission_Type','Registration Status':'Registration_Status'},inplace=True)

data_set.drop(['Model Year','Unnamed: 0','Model Name','Location','Body Type','Color'], axis=1, inplace=True)
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
x = data_set.drop(['Company_Name','Price'],axis=1)
y = data_set['Price']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.20, random_state=0)
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("Prediction Result")
print(df2.to_string())
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

