#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------------PRE PROCESSING-----------------------------------------i----

#Importing the dataset
dataset = pd.read_csv('Life Expectancy Data.csv')

#Print the head of dataset
print('data shape:',dataset.shape)
print('**************************************')
print('data head = \n' , dataset.head())
print('------------------------------------------------------------------------')

# Check non values
print("number nan value: \n" , dataset.isna().sum())
print('------------------------------------------------------------------------')

#Drop col
dataset.drop(['Hepatitis B','GDP','Population'],inplace=True,axis=1)
print(dataset.columns.tolist())
print('------------------------------------------------------------------------')

#Handel the nan values
nan = []
for column in dataset.columns:
    if dataset[column].isna().sum():
        nan.append(column)

print("Nan columns:\n" , nan)
print('------------------------------------------------------------------------')

#Fill missing values with mean
for column in nan:
    dataset[column].fillna(dataset[column].mode()[0] , inplace=True)

#Again! check non values
print("Check nan value: \n" , dataset.isna().sum())
print('------------------------------------------------------------------------')

#Encode the string columns
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dataset['Country'] = encoder.fit_transform(dataset['Country'])
dataset['Status'] = encoder.fit_transform(dataset['Status'])
print('After Encoding: = \n' , dataset.head())
print('------------------------------------------------------------------------')

#Spilt the target from the data
x = dataset.drop('Life expectancy ', axis=1)
y = dataset['Life expectancy ']
print('After Spilting (x): = \n' , x)
print('After Spilting (y): = \n' , y)
print('------------------------------------------------------------------------')

#Feature scaling by StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print('After scaling (x) by StandardScaler: = \n' , x)
print('------------------------------------------------------------------------')

#Feature scaling by MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
x = sc.fit_transform(x)
print('After scaling (x) by MinMaxScaler: = \n' , x)
print('------------------------------------------------------------------------')

#visual scalining
dataset.hist(figsize=(20,20))

#Spliting dataset into (Training set & Test set)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10,random_state=0)

#Linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#Predicting new results
y_pred = regressor.predict(x_test)

#Evaluate your model : R square and MAE / RMSE
from sklearn.metrics import mean_absolute_error ,mean_squared_error , r2_score
mea = mean_absolute_error(y_test , y_pred)
print(f'Mean Absolute error=   {mea:4f}\n')
mse = mean_squared_error(y_test , y_pred)
print(f'Mean Squared =   {mse:4f}\n')
r2 = r2_score(y_test , y_pred)
print(f'R2 Score =   {r2:4f}\n')
print('------------------------------------------------------------------------')

#SVR regression model
from sklearn.svm import SVR
srv = SVR()
srv.fit(x_train,y_train)

#Predicting new results
y_pred = srv.predict(x_test)

#Evaluate your model : R square and MAE / RMSE
from sklearn.metrics import mean_absolute_error ,mean_squared_error , r2_score
mea = mean_absolute_error(y_test , y_pred)
print(f'Mean Absolute error=   {mea:4f}\n')
mse = mean_squared_error(y_test , y_pred)
print(f'Mean Squared =   {mse:4f}\n')
r2 = r2_score(y_test , y_pred)
print(f'R2 Score =   {r2:4f}\n')
print('------------------------------------------------------------------------')

plt.figure(figsize=(10,8))
plt.scatter(y_test, y_pred, color='red', label='Prediction Vs Actual')
plt.plot(y_test , y_test , 'k--' , lw=2) #Diagonal line for reference
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()
