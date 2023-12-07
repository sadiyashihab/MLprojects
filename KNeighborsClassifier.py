import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay , classification_report , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

data =pd.read_csv('Healthcare-Diabetes.csv')
print(data.head())
print(data.tail())
print(data.info())
print(data.shape)
print(data.describe())
x = data.drop(["Outcome"],axis=1)
y = data["Outcome"]
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25,random_state=42,stratify = y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
lin = LinearRegression()
lin.fit(x_train,y_train)
print(lin.score(x_train,y_train))
print(lin.score(x_test,y_test))
lr = LogisticRegression(max_iter=500)
lr.fit(x_train,y_train)
print(lr.score(x_train,y_train))
print(lr.score(x_test,y_test))
kn = KNeighborsClassifier(n_neighbors=2)
kn.fit(x_train,y_train)
print(kn.score(x_train,y_train))
print(kn.score(x_test,y_test))
from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier()
classifier.fit(x_train, y_train)
y_pred= classifier.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("Prediction Result")
print(df2.to_string())
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))