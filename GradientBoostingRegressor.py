'''Author: Anandita Ashwath
        Description:Gradient Boosting Regressor
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
from sklearn.datasets import make_classification
from sklearn import ensemble        # library of Gradient Boosting regression
from sklearn.model_selection import train_test_split    # split data to training set and tesing set
from sklearn.metrics import mean_squared_error,mean_squared_log_error      # calculate MSE
from sklearn.externals import joblib    # for saving and loading model
from sklearn.metrics import accuracy_score #accuracy
from sklearn.inspection import permutation_importance


random= os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/gb_df.csv')
gb_df= pd.read_csv(random,delimiter=",")

path = os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/BikeClean/df.csv')
all_data= pd.read_csv(path,delimiter=",",parse_dates=["Date"])
all_data = all_data[(all_data["Date"] >= "2019-05-01") & (all_data["Date"] <= "2020-05-31")].reset_index(drop=True)

le_season = preprocessing.LabelEncoder()
gb_df["Season Code"] = le_season.fit_transform(gb_df["Season"])
le_time = preprocessing.LabelEncoder()
gb_df["Time Code"] = le_time.fit_transform(gb_df["Time"])

test_size = 0.30


X = gb_df[Help.CONSIDERING_FACTORS].copy()
y = gb_df[Help.PREDICTING_FACTOR].copy()
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=test_size, random_state=7)

params = {'n_estimators': 500,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.1,
          'loss': 'ls'}

reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

mse = mean_squared_error(y_test, reg.predict(X_test))
rmse = math.sqrt(mse)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))


#Plotting training deviance

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()

#Plotting feature importance
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.title('Feature Importance (MDI)')

result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=X.columns[sorted_idx])
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()


