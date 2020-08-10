import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import pandas as pd


def get_station_df():
    df = pd.read_csv("/Users/TEMP.ANANDITA.000/Desktop/Station_details.csv")
    df = df[df.columns[1:]]
    df[['Latitude']] = (df[['Latitude']]).astype(float) / 10**6
    df[['Longitude']] = (df[['Longitude']]).astype(float) / 10**6
    return df

station_df = get_station_df()
print(station_df[['Latitude']])
img = imread("/Users/TEMP.ANANDITA.000/Desktop/map.png")
border = [ -6.3504, -6.2189, 53.3206, 53.3761]
area = 1265.0/893.0   
stations = 118
xx = [np.random.uniform(border[0],border[1]) for x in range(stations)]
yy = [np.random.uniform(border[2],border[3]) for x in range(stations)]
print (xx)
print (yy)
plt.imshow(img,zorder=0, extent=border, aspect = area)
plt.plot(xx, yy, 'bo')
plt.show()
