'''Author:Anandita Ashwath
   Description: Random Forest Classification
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import ensemble        # library of RandomForest
from sklearn.model_selection import train_test_split    # split data to training set and testing set
from sklearn.metrics import mean_squared_error,mean_squared_log_error      # calculate MSE
from sklearn.externals import joblib    # for saving and loading model
from sklearn.metrics import accuracy_score,confusion_matrix

random= os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/gb_df.csv')
gb_df= pd.read_csv(random,delimiter=",")

path = os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/BikeClean/df.csv')
all_data= pd.read_csv(path,delimiter=",",parse_dates=["Date"])
all_data = all_data[(all_data["Date"] >= "2019-05-01") & (all_data["Date"] <= "2020-05-31")].reset_index(drop=True)

le_season = preprocessing.LabelEncoder()
gb_df["Season Code"] = le_season.fit_transform(gb_df["Season"])
le_time = preprocessing.LabelEncoder()
gb_df["Time Code"] = le_time.fit_transform(gb_df["Time"])
######################################################################
######### TRAINING MODEL USING RANDOM FOREST ALGORITHM ###########
######################################################################
# Create training and testing samples with 70% for training set, 30% for testing set using library

test_size = 0.30

model = RandomForestClassifier(n_estimators=100,random_state=7)
x = gb_df[Help.CONSIDERING_FACTORS].copy()
y = gb_df[Help.PREDICTING_FACTOR].copy()
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=test_size, random_state=7)


# feed training data Random Forest model
model.fit(x_train, y_train)

predictions=model.predict(x_test)

mse = mean_squared_error(y_test, model.predict(x_test))
rmse = math.sqrt(mse)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The root mean squared error (RMSE) on test set: {:.4f}".format(rmse))
print("Accuracy:",accuracy_score(y_test, predictions))


# Plot feature importance
feature_importance = model.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig, ax = plt.subplots()
ax.barh(pos, feature_importance[sorted_idx], align='center')
ax.set(title = 'Variable Importance',xlabel = 'Relative Importance')
plt.yticks(pos, x.columns[sorted_idx])
# set margins
plt.subplots_adjust(left=0.2, right=0.9, top=0.95, bottom=0.1)
fig.savefig("Randomforest_feature_importance.png")
fig.clear()



test = pd.DataFrame(x_test, columns=Help.CONSIDERING_FACTORS)
test["Time"] = le_time.inverse_transform(test["Time Code"].astype(np.int64))
test = test.drop(["Time Code"], axis = 1)
test[Help.PREDICTING_FACTOR] = y_test
test["pred"] = predictions.round(0).astype(np.int64)

test = pd.merge(test
                    , gb_df[["Number", "Address", "Bike Stands", "Latitude", "Longitude", "Time"]]
                    , how="left"
                    , on=["Latitude", "Longitude", "Time"])
test = test.groupby(["Number", "Address", "Time"]).agg({Help.PREDICTING_FACTOR: "mean", "pred": "mean", "Bike Stands": "max"}).reset_index()
#print(test.dtypes)

# get station numbers in testing set
station_numbers = test["Number"].unique()
print("Station numbers in testing set: " ,station_numbers)
# calculate number of stations in testing set
n_stations = len(station_numbers)
n_station_row = round(n_stations / Help.MAX_AXES_ROW)
n_station_row = n_station_row + 1 if n_station_row * Help.MAX_AXES_ROW < n_stations else n_station_row
print(f"We need to generate a figure with {n_station_row} rows for {n_stations}")

# ignore data from 00:00:00 to 05:30:00 since Dublin Bikes system doesn't operate in that time period
test = test[(test["Time"] >= "05:00:00")].reset_index(drop=True)
index = 0
fig, axes = plt.subplots(figsize = (12, 10), nrows = n_station_row, ncols = Help.MAX_AXES_ROW, sharex = True, sharey= True, constrained_layout=False)
for row in axes:
    for ax in row:
        #print(f"Rendering in {index}")
        if index >= n_stations:
            # locate sticks every 1 hour
            ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
            # show locate label with hour and minute format
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # set smaller size for tick labels
            ax.xaxis.set_tick_params(labelsize=7)
            # increase index of next station by 1 before continuing
            index += 1
            continue
        condition = test["Number"] == station_numbers[index]
        ax_x = pd.to_datetime(test[condition]["Time"], format="%H:%M:%S")
        ax_y1 = test[condition][Help.PREDICTING_FACTOR]
        ax_y2 = test[condition]["pred"]
        ax_y3 = test[condition]["Bike Stands"]
        ax.plot(ax_x, ax_y1, "b-", label='Actual')
        ax.plot(ax_x, ax_y2, "r-", label='Predicted')
        ax.plot(ax_x, ax_y3, "-.", color = 'black', label='Bike Stands')
        ax.fill_between(ax_x.dt.to_pydatetime(), ax_y2 - rmse, ax_y2 + rmse, facecolor='#3a3a3a', alpha=0.5)
        y_min = 0
        y_max = all_data["Bike Stands"].max()
        ax.set_ylim([y_min, y_max])
        # locate sticks every 1 hour
        ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
        # show locate label with hour and minute format
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        # set smaller size for tick labels
        ax.xaxis.set_tick_params(labelsize=7)
        # set title for each axe
        ax_title = test[condition]["Address"].unique()[0]
        ax.set_title(ax_title)
        # margin x at 0 and y at 0.1
        ax.margins(x=0.0, y=0.1)
        ax.grid(linestyle="-")
        # increase index of next station by 1
        index += 1
        handles, labels = ax.get_legend_handles_labels()

# show rotate tick lables automatically with 90 degree
fig.autofmt_xdate(rotation = "90")
# set title of the figure
fig.suptitle("Random Forest  prediction and actual number")
fig.subplots_adjust(hspace=0.6)
# Set Help labels
fig.text(0.5, 0.12, "Time", ha='center', va='center', fontsize="medium")
fig.text(0.06, 0.5, "Mean Available Stands", ha='center', va='center', rotation='vertical', fontsize="medium")
# plot the legend
fig.legend(handles, labels, title="Color", loc='center', bbox_to_anchor=(0.5, 0.06, 0., 0.), ncol=4)
fig.savefig("Randomforest_prediction_Prediction_new.png")
fig.clear()

