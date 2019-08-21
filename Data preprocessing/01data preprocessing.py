# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 18:26:20 2018

@author: sethu

"""
# Importing libraries

import numpy as np
import pandas as pd
import matplotlib as plt

# Importing data and splitting

dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

# Replacing missing values with mean

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

# Encoding categorical data for independent variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,0]=labelencoder_x.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

# Encoding categorical data for dependent variable
labelencoder_y=LabelEncoder()
Y=labelencoder_y.fit_transform(Y)

#splitting the data as training data and test data
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)




#






