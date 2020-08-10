'''Author:Anandita Ashwath
'''

import os
import numpy as np
import pandas as pd
import time
from helper import Help
import sys
import math
import fnmatch
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.externals import joblib    # for saving and loading model
from sklearn import preprocessing   # label encoder
from sklearn.metrics import mean_squared_error,mean_squared_log_error      # calculate MSE

start = time.time()

Help.create_folder(Help.EVALUATION_PLOTS_DIR)
Help.create_folder(Help.UNSEEN_PREDICTING_PLOTS_DIR)

def setStandstands(number):
    if (number == 79):
        total_stands = 27
    elif (number == 5):
        total_stands = 40
    elif (number == 100):
        total_stands = 25
    elif (number == 66):
        total_stands = 40
    else:
        total_stands = 23
    return total_stands

def calcStands(tota_stands_amount, stands_amount, Stands_amount):
    if (np.isnan(Stands_amount)):
        return Stands_amount
    else:
        return tota_stands_amount - stands_amount

# get the current working directory
working_dir = os.getcwd()

unseen_gb_df =Help.get_dataframe_from_file(Help.GRADIENT, True)
unseen_gb_df = unseen_gb_df[(unseen_gb_df["Time"] >= "05:00:00")].reset_index(drop=True)
# convert time string to time to plot data
unseen_gb_df["Time"] = pd.to_datetime(unseen_gb_df["Time"], format="%H:%M:%S")

# get station numbers in unseen set
station_numbers = unseen_gb_df["Number"].unique()
print("Station numbers in unseen set: ", station_numbers)
# calculate number of stations in testing set
n_stations = len(station_numbers)

# load Gradient Boosting model
model = joblib.load(Help.GRADIENT_BOOSTING_MODEL_FULL_PATH)

##################################################################
######################## PREDICT UNSEEN DATA #####################
##################################################################
# add predicted result into unseen dataset
unseen_gb_df["pred"] = model.predict(unseen_gb_df[Help.IMPORTANT_FACTORS]).round(0).astype(np.int64)
unseen_gb_df['pred'] = abs(unseen_gb_df['pred'])
#unseen_gb_df['pred'] = unseen_gb_df['pred'].clip(lower=0)
unseen_gb_df["diff"] = abs(unseen_gb_df[Help.PREDICTING_FACTOR] - unseen_gb_df["pred"])

Help.save_csv(unseen_gb_df, "./unseen_gb_df.csv")

mse = mean_squared_error(unseen_gb_df[Help.PREDICTING_FACTOR], unseen_gb_df["pred"])
rmse = math.sqrt(mse)
rmsle= mean_squared_log_error(unseen_gb_df[Help.PREDICTING_FACTOR], unseen_gb_df["pred"])
print("MSE: %.4f" % mse)
print("RMSE: %.4f" % rmse)
print("RMSLE : %.4f" % rmsle)


unseen_gb_df = unseen_gb_df.groupby(["Number", "Address", "Time", "Weekday"]).agg({Help.PREDICTING_FACTOR: "mean", "pred": "mean", "Bike Stands": "max"}).reset_index()
unseen_gb_df[Help.PREDICTING_FACTOR] = unseen_gb_df[Help.PREDICTING_FACTOR].round(0).astype(np.int64)
unseen_gb_df["pred"] = unseen_gb_df["pred"].round(0).astype(np.int64)
#Help.save_csv(unseen_gb_df, "./unseen_gb_df.csv")
# plot by weekdays
n_wdays = len(Help.SHORT_WEEKDAY_ORDER)
n_wday_row = round(n_wdays / Help.MAX_AXES_ROW)
n_wday_row = n_wday_row + 1 if (n_wdays % Help.MAX_AXES_ROW) > 0 else n_wday_row
for i in range(1, Help.MAX_STATION_NUMBER + 1):
    index = 0
    fig, axes = plt.subplots(figsize = (8, 7), nrows = n_wday_row, ncols = Help.MAX_AXES_ROW, sharex = True, sharey= True, constrained_layout=False)
    # real index of current station in array
    j = -1
    try:
        j = station_numbers.tolist().index(i)
    except:
        print(f"Not found {i}")
    if j == -1:
        continue
    fig_title = unseen_gb_df[unseen_gb_df.Number == station_numbers[j]].Address.unique()[0]
    # no data of current station number, move to the next station number
    if (len(unseen_gb_df[unseen_gb_df.Number == station_numbers[j]]) <= 0):
        print(f"No data for plotting station {fig_title}")
        continue
    for row in axes:
        for ax in row:
            if index >= n_wdays:
                # locate sticks every 1 hour
                ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
                # show locate label with hour and minute format
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                # set smaller size for tick labels
                ax.xaxis.set_tick_params(labelsize=7)
                # increase index of next station by 1 before continuing
                index += 1
                continue
            condition = (unseen_gb_df["Number"] == station_numbers[j]) & (unseen_gb_df["Weekday"] == Help.SHORT_WEEKDAY_ORDER[index])
            ax_x = unseen_gb_df[condition]["Time"]
            # actual Stands of its station
            ax_y1 = unseen_gb_df[condition][Help.PREDICTING_FACTOR]
            # predict Stands of its station
            ax_y2 = unseen_gb_df[condition]["pred"]
            # total bike stands of its station
            ax_y3 = unseen_gb_df[condition]["Bike Stands"]
            ax.plot(ax_x, ax_y1, "b-", label=u'Actual')
            ax.plot(ax_x, ax_y2, "r-", label=u'Predicted')
            ax.plot(ax_x, ax_y3, "-.", color = 'black', label=u'Bike Stands')
            #print(ax_x.dtypes)
            ax.fill_between(ax_x.dt.to_pydatetime(), ax_y2 - rmse, ax_y2 + rmse, facecolor='#3a3a3a', alpha=0.5)
            y_min = 0
            y_max = unseen_gb_df["Bike Stands"].max()
            ax.set_ylim([y_min, y_max])
            # locate sticks every 1 hour
            ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
            # show locate label with hour and minute format
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            # set smaller size for tick labels
            ax.xaxis.set_tick_params(labelsize=7)
            # set title for each axe
            ax_title = unseen_gb_df[condition]["Weekday"].unique()[0]
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
    fig.suptitle(f"Predictions for unseen data in {fig_title} from week of 01/05/2019   ")
    fig.subplots_adjust(hspace=0.2)
    # Set Help labels
    fig.text(0.5, 0.12, "Time", ha='center', va='center', fontsize="medium")
    fig.text(0.06, 0.5, "Mean Available Stands", ha='center', va='center', rotation='vertical', fontsize="medium")
    # plot the legend
    fig.legend(handles, labels, title="Color", loc='center', bbox_to_anchor=(0.5, 0.06, 0., 0.), ncol=4)
    fig.savefig(f"{Help.PREDICTING_PLOTS_DIR}/unseen/unseen_prediction_{i}.png")
    plt.close()

