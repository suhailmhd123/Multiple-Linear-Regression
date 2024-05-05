# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:40:59 2024

@author: H P
"""
# gradient descent rigression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stp=pd.read_csv(r"C:\SUHAIL\Reshma miss\50_Startups.csv")
stp
stp.describe(include="all")
stp.shape
stp.columns
stp.isna().sum()

# using simple imputer 
from sklearn.impute import SimpleImputer
values = SimpleImputer(missing_values=np.nan,strategy="mean")
stp["Administration"] = pd.DataFrame(values.fit_transform(stp[["Administration"]]))
stp["Administration"].isna().sum()

# encoding
stp.State.unique()
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
stp["State"] = labelencoder.fit_transform(stp["State"])

stp.skew()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Split the data into train and test sets
x = stp[['RD', 'Administration', 'Marketing', 'State']]
y = stp['Profit']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)
x_train.shape,y_train.shape,x_test.shape,y_test.shape

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)


# Predict on the test set
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)
y_pred_test
y_pred_train

# Evaluate the model
mse = mean_absolute_error(y_test,y_pred_test)
print("Mean squared error:", mse)
rmse_test = np.sqrt(mse)
rmse_test

mse1 = mean_absolute_error(y_train,y_pred_train)
print("Mean squared error:", mse1)
rmse_train = np.sqrt(mse1)
rmse_train

# Check accuracy scores
accu_train = model.score(x_train, y_train)
accu_train
accu_test = model.score(x_test, y_test)
accu_test

#####multicollinearity

#correlation
stp.corr()

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Select your independent variables (features) for the model
X = add_constant(x_train)

# Calculate VIF
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Filter out the constant term
vif = vif[vif["Variable"] != "const"]

print(vif)

##Lasso regression
from sklearn.linear_model import Lasso

# initialize Lasso Regression Model
lasso_model = Lasso(alpha=0.1) # You can adjust the alpha parameter for regularization strength
 
# Fit the Model
lasso_model.fit(x_train,y_train)

# Predictions
y_pred_lasso = lasso_model.predict(x_test)

# Evaluate the Model
mse_lasso = mean_absolute_error(y_test, y_pred_lasso)
print("Mean squared error:", mse_lasso)
rmse_l = np.sqrt(mse_lasso)
rmse_l

## Ridge Regression
from sklearn.linear_model import Ridge

# Initialize Ridge Regression Model
ridge_model = Ridge(alpha=0.1)

# Fit the Model
ridge_model.fit(x_train,y_train)
# Predictions
y_pred_ridge = ridge_model.predict(x_test)

# Evaluate the Model
mse_ridge = mean_absolute_error(y_test,y_pred_ridge)
print("Mean squared error:", mse_ridge)
rmse_ridge = np.sqrt(mse_ridge)
rmse_ridge

## Elastic net Model
from sklearn.linear_model import ElasticNet
 
# Initialize Elastinet Regression Model
elastic_net_model = ElasticNet(alpha=0.1)

# Fit hte Model
elastic_net_model.fit(x_train,y_train)

# Predictions
y_pred_elastic_net = elastic_net_model.predict(x_test)

# Evaluate the Model
mse_enet = mean_absolute_error(y_test,y_pred_elastic_net)
print("Mean squared error:", mse_enet)
rmse_enet = np.sqrt(mse_enet)
rmse_enet

# Storing regularization values in a data frame
d1 = {"Regularizations":["Lasso", "Ridge", "ElasticNet"], "RMSE":[rmse_l,rmse_ridge,rmse_enet]}
rmse_frame = pd.DataFrame(d1)
rmse_frame












