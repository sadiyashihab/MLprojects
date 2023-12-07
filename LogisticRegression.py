import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
df = pd.read_csv('titanic_survived.csv')
print(df.head())
print(df.tail())
print(df.info())
print(df.shape)
print(df.isnull().sum())
print(df.duplicated().sum())
ohe_columns = ['Age', 'Sex']
ohe = OneHotEncoder(sparse=False).fit(df[ohe_columns])
encoded = ohe.transform(df[ohe_columns])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())
df = pd.concat([df[[x for x in df.columns if x not in ohe_columns]].reset_index(drop=True),encoded_df], axis=1)
df.drop(['PassengerId', 'Ticket', 'Name', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)
columns = [c for c in df.columns if c != 'Survived']
X = df[columns]
Y = df['Survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
log = LogisticRegression()
log.fit(X_train, Y_train)
accuracy = accuracy_score(Y_test, log.predict(X_test))
precision = precision_score(Y_test, log.predict(X_test))
recall = recall_score(Y_test, log.predict(X_test))
f1 = f1_score(Y_test, log.predict(X_test))
conf_matrix = confusion_matrix(Y_test, log.predict(X_test))
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", conf_matrix)
