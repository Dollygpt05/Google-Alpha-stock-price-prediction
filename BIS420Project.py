
# Import pandas to read csv file

import pandas as pd

alpha=pd.read_csv('GOOGLE.csv')
print(alpha.head())

# To get number of training days
print("trainingdays=", alpha.shape)

# To visualize close price data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('pastel')
plt.figure(figsize=(10,4))
plt.title("Google's Stock Price")
plt.xlabel("Days")
plt.ylabel("Closed Price USD")
plt.plot(alpha['Close'])
plt.show()

# To get close price
alpha=alpha[['Close']]
print(alpha)

# Creating variable for 'X'Days in future

futuredays=30

# To create a new unit X for units/day
alpha['prediction']=alpha[['Close']].shift(-futuredays)
print(alpha.head())
print(alpha.tail())

# To create a data set to convert into numpy array
import numpy as np
x = np.array(alpha.drop(['prediction'], axis=1))[:-futuredays]
print(x)
y=np.array(alpha['prediction'])[:-futuredays]
print(y)

# To devide training and testing data

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.25)

# To create a model
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor().fit(xtrain, ytrain)

from sklearn.linear_model import LinearRegression
linear=LinearRegression().fit(xtrain,ytrain)

xfuture = alpha.drop(['prediction'],axis=1)[:-futuredays]
xfuture=xfuture.tail()
xfuture=np.array(xfuture)
print(xfuture)

# Decision tree prediction
treeprediction=tree.predict(xfuture)
print("Decision tree prediction:",treeprediction)

# Linear prediction
linearprediction=linear.predict(xfuture)
print("Linear prediction:",linearprediction)

# To visualize decision tree prediction

prediction = treeprediction
valid=alpha[x.shape[0]:]
plt.figure(figsize=(10,6))
plt.title("Google Stock Price Prediction(Decision Tree)")
plt.xlabel('Days')
plt.ylabel('Close price USD')
plt.plot(alpha['Close'])
plt.plot(valid[['Close']])
plt.legend(["Original","Valid","Prediction"])
plt.show()

# To visualize linear prediction

prediction=linearprediction
valid=alpha[x.shape[0]:]
plt.figure(figsize=(10,6))
plt.title("Google Stock Price Prediction(Linear regression)")
plt.xlabel('Days')
plt.ylabel('Close price USD')
plt.plot(alpha['Close'])
plt.plot(valid[['Close']])
plt.legend(["Original","Valid","Prediction"])
plt.show()

# To validate the training model

from sklearn.metrics import mean_absolute_error

# To Calculate MAE for Decision Tree model
tree_mae = mean_absolute_error(ytest[:len(treeprediction)], treeprediction)
print("Mean Absolute Error (Decision Tree):", tree_mae)

# To Calculate MAE for Linear Regression model
linear_mae = mean_absolute_error(ytest[:len(linearprediction)], linearprediction)
print("Mean Absolute Error (Linear Regression):", linear_mae)


import numpy as np

# To Calculate squared errors
tree_squared_error = np.mean((ytest[:len(treeprediction)] - treeprediction) ** 2)
linear_squared_error = np.mean((ytest[:len(linearprediction)] - linearprediction) ** 2)

# To Calculate RMSE
tree_rmse = np.sqrt(tree_squared_error)
linear_rmse = np.sqrt(linear_squared_error)

print("Root Mean Squared Error (Decision Tree):", tree_rmse)
print("Root Mean Squared Error (Linear Regression):", linear_rmse)
