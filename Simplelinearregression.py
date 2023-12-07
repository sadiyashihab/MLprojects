import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('insurance_data.csv')
print('shape: ', data.shape)
# preparing data for model
x = data['Age'].values
x = x.reshape(-1, 1)
y= data['Premium'].values
y = y.reshape(-1, 1)
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
# building and applying linear regression model
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the results
print("#" * 40)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")
print("#" * 40)
