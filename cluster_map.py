'''Author: Anandita Ashwath
    Description:
        Plotting the clustered stations on to a Map using Folium map an interactive map leaflet
        Along with station coordinates from station details.The rendered map is saved to an HTML format 
'''


import pandas as pd
import os
import datetime
from helper import Help
import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import seaborn as sns
import folium


#Reading clustered station data
cluster=Help.get_dataframe_from_file(Help.CLUSTERED_DATA_FILE_FULL_PATH, True)
path = os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/Station_details.csv')

#Reading station details
station= pd.read_csv(path,delimiter=",")
station=station.set_index("Name")
merged = pd.merge(station
                ,cluster[["Number","Name","Cluster"]]
                , on=["Number", "Name"]
                , how="left")


merged=merged.dropna()
print(merged)

#Assigning clusters a color each and rendering the map.
n_clusters=4
colours = ['red','blue','green','orange']
mp = folium.Map(location=[53.34, -6.2603], zoom_start=14,tiles='cartodbpositron')
for c , colour in zip(range(1,n_clusters+1),colours):
    tmp = merged[merged['Cluster'] == c]
#     for 
    for location in tmp.iterrows():
            folium.CircleMarker(
                location=[location[1]['Latitude'],location[1]['Longitude']],
                radius=7,
                popup=location[0],
                color=colour,
                fill_color=colour
            ).add_to(mp)

#Saving the map to html format
mp.save('map.html')
mp
