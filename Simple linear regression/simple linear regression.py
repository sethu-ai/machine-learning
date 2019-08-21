# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:08:06 2018

@author: sethu
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1]

#splitting the data for training and test
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)

#fitting simple linear regression to the training set 
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#prediciting the test set results
y_pred=regressor.predict(X_test)

#visualising the training_set results
plt.scatter(X_train,y_train,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('SalaryVsExperience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

#visualising the training_set results
plt.scatter(X_test,y_test,color='Red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('SalaryVsExperience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()
