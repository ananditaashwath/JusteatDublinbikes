'''Author: Anandita Ashwath
    Description:
       Clustering Analysis. Adapted from the code written by Tam M Pham. 
'''

import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import sys

def fill_nan(df):
    # iterate through rows
    for i, row in df.iterrows():
        # iterate through columns of the current row
        for j, column in row.iteritems():
            # if the current row is the first row and it has N/A value, fill in with the next non-N/A value
            if (i == 0 and np.isnan(df.loc[i,j])):
                k = i+1
                # iterate to find the next non-N/A value
                while (np.isnan(df.loc[k,j])):
                    k = k+1
                df.at[i,j] = df.at[k,j]
            elif (np.isnan(df.at[i,j])):    # if the current row is other rows and it has N/A value, fill in with the previous value
              df.at[i,j] = df.at[i-1,j]
            else:   # keep its value
                df.at[i,j] = df.at[i,j]
    return df


def time(df):
    # group data by Number, Date and Time
    df = df.set_index(["Number", "Date", "Time"])["Available Stands"]  
    #print(df)
    # then unstack the data frame, reset index
    df = df.unstack().reset_index()
    "Unstack the dataframe"
    #print(df)
    df = df.rename_axis(None, axis=1)
    # drop columns Number and Date`
    df = df.drop(["Number", "Date"], axis=1)
    df = fill_nan(df)
    "Fill N/A values"
    #print(df)
    return df

def cal_wss(df):
    row_len = len(df.index)
    wss = []    # within-cluster sum of squares
    wss.append((row_len-1) * df.var().sum())   # this equation comes from http://statmethods.net/advstats/cluster.html
    for i in range(2, 15):
        clusters = KMeans(i).fit(df)
        wss.append(clusters.inertia_)
    return wss

def kmeans(df, cluster_no, time_lvls):
    # initialize K-means and fit the input data
    kmeansfit = KMeans(n_clusters=cluster_no).fit(df)
    # predict the clusters
    labels = kmeansfit.predict(df)
    # get the cluster centroids_df
    centroids = kmeansfit.cluster_centers_
    # format cluster centroids_df to dateframe for viewing
    centroids_df = pd.DataFrame(centroids, columns=time_lvls)
    centroids_df["Cluster"] = centroids_df.index + 1
    # reshape the data for plotting
    centroids_df = pd.melt(centroids_df, id_vars=["Cluster"], var_name="Time", value_name="Available Stands")
    centroids_df["Time"] = pd.to_datetime(centroids_df["Time"], format="%H:%M:%S")
    return centroids_df

path = os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/DublinBikes/df.csv')
df= pd.read_csv(path,delimiter=",",parse_dates=["Date"])


# clone data frame from df.csv to another dataframe for pre-processing
prep = df[["Number", "Date", "Time", "Available Stands"]].copy()
# spread the data
prep = time(prep)

# the time levels is used for reshaping the data
time_lvls = df[["Time"]].drop_duplicates(keep='first').sort_values(by=["Time"]).reset_index(drop=True)["Time"]

wss = cal_wss(prep)

clusters_df = pd.DataFrame({"num_clusters": range(1, 15), "wss": wss})
fig, ax = plt.subplots(figsize=(6,5))
ax.plot(clusters_df.num_clusters, clusters_df.wss, marker = "o" )
ax.set(title = "Optimal number of clusters using Elbow method", xlabel="Number of Clusters", ylabel="Total Within Sum of Squares")
fig.savefig("wss_all_stations.png")
fig.clear()

centroids_df = kmeans(prep,4, time_lvls)
print(centroids_df)

cluster_centers = centroids_df.copy()
cluster_centers =cluster_centers.groupby(["Cluster"])["Available Stands"].sum().reset_index()
cluster_centers =cluster_centers.rename(columns={"Available Stands": "Sum Of Stands"})
print(cluster_centers)

aggregate = df[["Number", "Name", "Time", "Available Stands"]].copy()
aggregate = aggregate.groupby(["Number", "Name", "Time"])["Available Stands"].mean().reset_index()
aggregate = aggregate.rename(columns={"Available Stands": "Stands"})
aggregate["Cluster"] = "None"
# Loop through each station and find the cluster which is closer to it
for i in range(1,118):
    min_diff = cluster_centers.copy()
    # Get the sum of stands for each station
    min_diff["Station Sum"] = aggregate[aggregate["Number"] == i]["Stands"].sum()
    min_diff["Diff"] = np.abs(min_diff["Sum Of Stands"] - min_diff["Station Sum"])
    # cluster which has the minimum difference between sum of stands and available stands is the station cluster
    min_diff = min_diff.sort_values(by=["Diff"]).reset_index(drop=True).head(1)
    print(min_diff)
    aggregate.loc[aggregate["Number"] == i, "Cluster"] = min_diff["Cluster"].values[0]
    #print(aggregate[aggregate["Number"] == i])
aggregate.to_csv("clustered_stations.csv",sep=",",encoding='utf-8',index=False)
#print(aggregate)

#Plotting Kmeans for all stations
print("Plotting K-Means of all stations during the week")
# plot available bike stands based on cluster number
fig, ax = plt.subplots(figsize=(8, 7))
for label,cluster_df in centroids_df.groupby("Cluster"):
    ax.plot(cluster_df["Time"], cluster_df["Available Stands"], label=label)
# locate sticks every 1 hour
ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
# show locate label with hour and minute format
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# show rotate tick labels automatically with 30 degree
fig.autofmt_xdate()
# set title, xlabel  and ylabel of the figure
ax.set(title="Clustering for all stations", xlabel="Hour of day", ylabel="Available Stands")
# set location of legend
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 0.65), loc=2, borderaxespad=0.)
# show grid
ax.grid(linestyle="-")
# margin x at 0 and y at 0.1
ax.margins(x=0.0, y=0.1)
# set margins
plt.subplots_adjust(left=0.09, right=0.85, top=0.95, bottom=0.1)
# save the plot to file
fig.savefig("clustering_of_stations.png")
fig.clear()


weekdays = pd.Series(np.array(["Mon", "Tue", "Wed", "Thu", "Fri"]))
weekday_df = df[df["Weekday"].isin(weekdays)][["Number", "Date", "Time", "Available Stands"]]
# spread the data
weekday_df =time(weekday_df)
wss = cal_wss(weekday_df)

clusters_df = pd.DataFrame({"num_clusters": range(1, 15), "wss": wss})
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(clusters_df.num_clusters, clusters_df.wss, marker = "o" )
ax.set(title = "Weekdays (Mon - Fri) optimal number of clusters", xlabel="Number of Clusters", ylabel="Total Within Sum of Squares")
fig.savefig("weekdays.png")
fig.clear()

centroids_df = kmeans(weekday_df,4, time_lvls)


print("Plotting K-Means of all stations during weekdays")

# plot available bike stands based on cluster number
fig, ax = plt.subplots(figsize=(8, 7))
for label,cluster_df in centroids_df.groupby("Cluster"):
    ax.plot(cluster_df["Time"], cluster_df["Available Stands"], label=label)
# locate sticks every 1 hour
ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
# show locate label with hour and minute format
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# show rotate tick lables automatically with 30 degree
fig.autofmt_xdate()
# set title, xlable and ylabel of the figure
ax.set(title="Weekdays Clustering (Mon - Fri)", xlabel="Hour of day", ylabel="Available Stands")
# set location of legend
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 0.65), loc=2, borderaxespad=0.)
# show grid
ax.grid(linestyle="-")
# margin x at 0 and y at 0.1
ax.margins(x=0.0, y=0.1)
# set margins
plt.subplots_adjust(left=0.09, right=0.85, top=0.95, bottom=0.1)
# save the plot to file
fig.savefig("clustering_weekdays.png")
fig.clear()

#Weekends
weekend_df = df[~df["Weekday"].isin(weekdays)][["Number", "Date", "Time", "Available Stands"]]
# spread the data
weekend_df = time(weekend_df)
wss = cal_wss(weekend_df)

clusters_df = pd.DataFrame({"num_clusters": range(1, 15), "wss": wss})
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(clusters_df.num_clusters, clusters_df.wss, marker = "o" )
ax.set(title = "Weekends (Sat - Sun) optimal number of clusters", xlabel="Number of Clusters", ylabel="Total Within Sum of Squares")
fig.savefig("weekends.png")
fig.clear()

centroids_df = kmeans(weekend_df, 4, time_lvls)

print("Plotting K-Means of all stations during weekends")
# plot available bike stands based on cluster number
fig, ax = plt.subplots(figsize=(8, 7))
for label, cluster_df in centroids_df.groupby("Cluster"):
    ax.plot(cluster_df["Time"], cluster_df["Available Stands"], label=label)
# locate sticks every 1 hour
ax.xaxis.set_major_locator(mdates.HourLocator(interval = 1))
# show locate label with hour and minute format
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# show rotate tick lables automatically with 30 degree, it would show 90 degree if we don't call it
fig.autofmt_xdate()
# set title, xlable and ylabel of the figure
ax.set(title="Weekends Clustering (Sat - Sun)", xlabel="Hour of day", ylabel="Available Stands")
# set location of legend
ax.legend(title="Cluster", bbox_to_anchor=(1.02, 0.65), loc=2, borderaxespad=0.)
# show grid
ax.grid(linestyle="-")
# margin x at 0 and y at 0.1
ax.margins(x=0.0, y=0.1)
# set margins
plt.subplots_adjust(left=0.09, right=0.85, top=0.95, bottom=0.1)
# save the plot to file
fig.savefig("clustering_weekends.png")
fig.clear()

