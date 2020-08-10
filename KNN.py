''' Author:Anandita Ashwath
    Description: K-nearest Neighbours regression
'''


import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import random
import datetime as dt
from helper import Help
from operator import is_not
from functools import partial
from sklearn import preprocessing   # label encoder
from sklearn.model_selection import train_test_split    # split data to training set and tesing set
from sklearn.metrics import mean_squared_error,mean_squared_log_error      # calculate MSE
from sklearn.metrics import accuracy_score #accuracy
from sklearn.preprocessing import StandardScaler #Scaling feature
from sklearn.neighbors import KNeighborsClassifier #for Knn 


#read the modelling data
random= os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/gb_df.csv')
gb_df= pd.read_csv(random,delimiter=",")


le_season = preprocessing.LabelEncoder()
gb_df["Season Code"] = le_season.fit_transform(gb_df["Season"])
le_time = preprocessing.LabelEncoder()
gb_df["Time Code"] = le_time.fit_transform(gb_df["Time"])


# Splitting the  training and testing  data 70:30
test_size = 0.30


X = gb_df[Help.CONSIDERING_FACTORS].copy()
y = gb_df[Help.PREDICTING_FACTOR].copy()
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=7)

#Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#knn classifier method implementation
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
print(accuracy_score(y_test,y_pred))


error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')