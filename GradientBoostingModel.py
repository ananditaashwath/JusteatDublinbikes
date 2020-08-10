'''Author:Anandita Ashwath
        Description Gradient Boosting Model -Classification prediction
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn import ensemble        # library of Gradient Boosting
from sklearn.model_selection import train_test_split    # split data to training set and tesing set
from sklearn.metrics import mean_squared_error,mean_squared_log_error      # calculate MSE
from sklearn.externals import joblib    # for saving and loading model
from sklearn.metrics import accuracy_score #accuracy

cluster_data= os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/clustered_stations.csv')
df= pd.read_csv(cluster_data,delimiter=",")

path = os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/DublinBikes/df.csv')
all_data= pd.read_csv(path,delimiter=",",parse_dates=["Date"])
all_data = all_data[(all_data["Date"] >= "2019-05-01") & (all_data["Date"] <= "2020-05-31")].reset_index(drop=True)

# left merge these two dataframes together based on Number, Date and Time
merged = pd.merge(all_data
                    ,df[["Number","Time", "Cluster"]]
                    , on=["Number", "Time"]
                    , how="left")

# Calculate activity in each cluster
cluster_activity = merged.copy()
cluster_activity["Activity"] = cluster_activity["Check In"] + cluster_activity["Check Out"]
cluster_activity = cluster_activity.groupby(["Number", "Cluster"])["Activity"].sum().reset_index(name="Total Activity")

# Find the most active station per cluster
top_stations = cluster_activity.copy()
top_stations = top_stations[top_stations.groupby(["Cluster"])["Total Activity"].transform(max) == top_stations["Total Activity"]].reset_index(drop=True)
print(top_stations)
# Find the least active station per cluster
low_stations = cluster_activity.copy()
low_stations = low_stations[low_stations.groupby(["Cluster"])["Total Activity"].transform(min) == low_stations["Total Activity"]].reset_index(drop=True)
print(low_stations)

# Turn the station number of the most active station and the least active station into a list
selected = top_stations["Number"].tolist() + low_stations["Number"].tolist()

# Randomly select other 3 stations in each cluster for the Gradient boosting modelling
for i in range(1,4+1):    # iterate throught from cluster 1 to cluster 4
    subset = merged[(merged["Cluster"] == i) & (~merged["Number"].isin(selected))].sample(n = 3)
    rand_list = subset["Number"].tolist()
    selected = selected + rand_list
print("Stations selected randomly is ", selected)

time_df = merged[merged["Number"].isin(selected)].copy()

# group time into 48 factors
time_df["Time"] = time_df["Time"].apply(lambda x:Help.refine_time(x))
time_df["Season"] = time_df["Date"].apply(lambda x:Help.define_season(x))
time_df[Help.PREDICTING_FACTOR] = time_df["Available Stands"]
time_df = time_df.groupby(["Number", "Name", "Address", "Date", "Time", "Bike Stands", "Weekday", "Season"]).agg({Help.PREDICTING_FACTOR: "mean", "Cluster": "first"}).reset_index()
time_df[Help.PREVIOUS_PREDICTING_FACTOR] = time_df.groupby(["Number", "Name", "Address", "Date"])[Help.PREDICTING_FACTOR].shift(1)
time_df[Help.PREVIOUS_PREDICTING_FACTOR] = time_df.apply(
    lambda row: row[Help.PREDICTING_FACTOR] if np.isnan(row[Help.PREVIOUS_PREDICTING_FACTOR]) else row[Help.PREVIOUS_PREDICTING_FACTOR],
    axis=1
)
# convert float64 columns to int64 columns, it converts numeric columns to float64
time_df[Help.PREDICTING_FACTOR] = time_df[Help.PREDICTING_FACTOR].astype(np.int64)
time_df[Help.PREVIOUS_PREDICTING_FACTOR] = time_df[Help.PREVIOUS_PREDICTING_FACTOR].astype(np.int64)

# read CSV file containing geographical info
geo =Help.get_dataframe_from_file("/Users/TEMP.ANANDITA.000/Desktop/Station_details.csv", True)
gb_df = pd.merge(time_df
                    , geo[["Number","Latitude", "Longitude"]]
                    , on=["Number"]
                    , how="left")

# read CSV file containing weather info
weather =Help.get_dataframe_from_file("/Users/TEMP.ANANDITA.000/Desktop/Weather.csv", True)
weather = weather.drop_duplicates(subset=["Station", "date","Rain","AirTemperature","AtmosphericPressure", "Windspeed"], keep='first')
weather["date"] = pd.to_datetime(weather["date"], format="%m/%d/%Y %H:%M")
weather["Date"] = pd.to_datetime(weather["date"].dt.strftime(Help.DATE_FORMAT),format=Help.DATE_FORMAT)
weather["Time"] = weather["date"].dt.strftime(Help.TIME_FORMAT)

# build important factors and formula to predict the bike number
gb_df = pd.merge(gb_df
                , weather[["Date", "Time","Rain","AirTemperature", "AtmosphericPressure", "Windspeed"]]
                , on=["Date", "Time"]
                , how="left")
gb_df["Rain"].fillna((gb_df["Rain"].mean()), inplace = True)  
gb_df["AirTemperature"].fillna((gb_df["AirTemperature"].mean()), inplace = True)              
gb_df["AtmosphericPressure"].fillna((gb_df["AtmosphericPressure"].mean()), inplace = True)
gb_df["Windspeed"].fillna((gb_df["Windspeed"].mean()), inplace = True)
gb_df["Weekday Code"] = pd.to_datetime(gb_df["Date"], format=Help.DATE_FORMAT).dt.weekday
# label encoding for weekdays, time and season
le_season = preprocessing.LabelEncoder()
gb_df["Season Code"] = le_season.fit_transform(gb_df["Season"])
le_time = preprocessing.LabelEncoder()
gb_df["Time Code"] = le_time.fit_transform(gb_df["Time"])
Help.save_csv(gb_df, "gb_df.csv")
#print(f"Data has {len(gb_df)} rows")

######################################################################
######### TRAINING MODEL USING GRADIENT BOOSTING ALGORITHM ###########
######################################################################
# Create training and testing samples with 70% for training set, 30% for testing set using library
seed = 7
test_size = 0.30
params = {'n_estimators': 500, 'max_depth': 4, 'learning_rate':0.1,'min_samples_split': 2, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)


x = gb_df[Help.CONSIDERING_FACTORS].copy()
y = gb_df[Help.PREDICTING_FACTOR].copy()
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=test_size, random_state=seed)


# feed training data to Gradient Boosting model
model.fit(x_train, y_train)

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
fig.savefig("feature_importance_gradient1.png")
fig.clear()

# after viewing feature importance, weather information doesn't impact the result
# so take it out
x = gb_df[Help.IMPORTANT_FACTORS].copy()
y = gb_df[Help.PREDICTING_FACTOR].copy()
x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=test_size, random_state=seed)

# feed training data to Gradient Boosting model
model.fit(x_train, y_train)

# save model
joblib.dump(model, Help.GRADIENT_BOOSTING_MODEL_FULL_PATH)

######################################################################
################ TESTING OUR GRADIENT BOOSTING MODEL #################
######################################################################
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
#rmsle=mean_squared_log_error(y_test,y_pred)
print("MSE: %.4f" % mse)
print("RMSE: %.4f" % rmse)
#print("RMSLE: %.4f" %rmsle)


test = pd.DataFrame(x_test, columns=Help.IMPORTANT_FACTORS)
test["Time"] = le_time.inverse_transform(test["Time Code"].astype(np.int64))
test = test.drop(["Time Code"], axis = 1)
test[Help.PREDICTING_FACTOR] = y_test
test["pred"] = y_pred.round(0).astype(np.int64)

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
fig.suptitle("Gradient Boosting prediction and actual number")
fig.subplots_adjust(hspace=0.6)
# Set Help labels
fig.text(0.5, 0.12, "Time", ha='center', va='center', fontsize="medium")
fig.text(0.06, 0.5, "Mean Available Stands", ha='center', va='center', rotation='vertical', fontsize="medium")
# plot the legend
fig.legend(handles, labels, title="Color", loc='center', bbox_to_anchor=(0.5, 0.06, 0., 0.), ncol=4)
fig.savefig("prediction_gradientboosting_new.png")
fig.clear()

