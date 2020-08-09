'''Author: Anandita Ashwath
    Description:
        Read raw data from multiple CSV files, then pre-processing the data 
        and finally store the data to file to re-use the data for exploring
        and clustering. Adapted from the code written by Tam M Pham. 
'''

import os
import glob
import numpy as np
import pandas as pd
import time

def count_check_in(diff):
    if (diff > 0):
        return diff
    return 0

def count_check_out(diff):
    if (diff < 0):
        return abs(diff)
    return 0

def refine_min(min):
    if min < 10: return "00"
    elif min < 20: return "10"
    elif min < 30: return "20"
    elif min < 40: return "30"
    elif min < 50: return "40"
    elif min < 60: return "50"
    else: 
        return np.nan

start = time.time()
#Reading all the csv files from the folder using glob moduleand then merging into one folder
os.chdir("/Users/TEMP.ANANDITA.000/Desktop/Dublinbikes")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
df = pd.concat([pd.read_csv(f) for f in all_filenames ])
os.chdir("/Users/TEMP.ANANDITA.000/Desktop/Dublinbikes")
print("Change back to root directory -> {0}".format(os.getcwd()))
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.dropna()
df = df.drop_duplicates(subset=["number", "last_update", "available_bike_stands"], keep='first')
df = df.reset_index(drop=True)
df["last_update"] = pd.to_datetime(df["last_update"], unit='ms', utc=True)
df["date"] = df["last_update"].dt.strftime('%Y-%m-%d')
df["time"] = df["last_update"].apply(lambda x: "%s:%s:00" % (x.strftime('%H'), refine_min(x.minute)))
df["weekday"] = df["last_update"].dt.strftime("%a")
df.sort_values(by=["number", "date", "time"], inplace=True)
df["last_available_stands"] = df.groupby(["number", "name", "address"])["available_bike_stands"].shift(1)
df["last_available_stands"] = df.apply(
    lambda row: row["available_bike_stands"] if np.isnan(row["last_available_stands"]) else row["last_available_stands"],
    axis=1
)
df["number"] = df.number.astype(np.int64)
df["bike_stands"] = df.bike_stands.astype(np.int64)
df["available_bike_stands"] = df.available_bike_stands.astype(np.int64)
df["last_available_stands"] = df.last_available_stands.astype(np.int64)
df["diff"] = df.available_bike_stands - df.last_available_stands
df["check_in"] = df["diff"].apply(lambda x: count_check_in(x))
df["check_out"] = df["diff"].apply(lambda x: count_check_out(x))
df = df.groupby(["number", "name", "address", "date", "time", "weekday"]).agg({"bike_stands": "min", 
        "diff": "sum", "available_bike_stands": "last", "check_in": "sum", "check_out": "sum"}).reset_index()
df = df.rename(columns={"number": "Number", "name": "Name", "address": "Address", "date": "Date", "time": "Time", "weekday": "Weekday", "bike_stands": "Bike Stands", "diff": "Diff", "available_bike_stands": "Available Stands", "check_in": "Check In", "check_out": "Check Out"})

df.to_csv( "df.csv", index=False, encoding='utf-8-sig')
end = time.time()
print("Done preparation after {} seconds".format((end - start)))


