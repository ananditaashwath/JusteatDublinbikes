'''Author: Anandita Ashwath
    Description:
         Merging all data along with weather details and holiday list.
         Exploring data based on bike usage for Air temperature 
         and Windspeed.Plotting based on average usage a day.

'''


import pandas as pd
import os
import datetime
from helper import Help
import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as mdates

#All data being read in, from May 2019 to May 2020
path = os.path.relpath('/Users/TEMP.ANANDITA.000/Desktop/DublinBikes/df.csv')
all_data= pd.read_csv(path,delimiter=",",parse_dates=["Date"])
all_data = all_data[(all_data["Date"] >= "2019-05-01") & (all_data["Date"] <= "2020-05-31")].reset_index(drop=True)

#Making a copy of the data in merged dataframe
merged=all_data.copy()

#Reading weather data 
weather =Help.get_dataframe_from_file("/Users/TEMP.ANANDITA.000/Desktop/Weather.csv", True)
weather = weather.drop_duplicates(subset=["Station", "date","Rain","AirTemperature","AtmosphericPressure", "Windspeed"], keep='first')
weather["date"] = pd.to_datetime(weather["date"], format="%m/%d/%Y %H:%M")
weather["Date"] = pd.to_datetime(weather["date"].dt.strftime(Help.DATE_FORMAT),format=Help.DATE_FORMAT)
weather["Time"] = weather["date"].dt.strftime(Help.TIME_FORMAT)

#merging weather data to all data
merged = pd.merge(all_data
                , weather[["Date", "Time","Rain","AirTemperature", "AtmosphericPressure", "Windspeed"]]
                , on=["Date", "Time"]
                , how="left")

merged=merged.dropna()
#print(merged)

#Reeading Holiday list 
holiday=Help.get_dataframe_from_file(Help.HOLIDAY_LIST, True)
holiday= holiday.drop_duplicates(subset=["Date","Holiday","Day"],keep='first')
holiday["Date"]=pd.to_datetime(holiday["Date"])
#holiday["Date"]=pd.to_datetime(holiday["Date"].dt.strftime(Help.DATE_FORMAT))

#print(holiday)

#Merging Holiday list with all data and weather data
merged=pd.merge(merged
               ,holiday[["Date","Holiday"]]
               ,on=["Date"]
               ,how="left")

#Creating a new column holidays to assign 'NaN' values to False and Not-Nan to 'True'
condition_one=(merged["Holiday"].isnull())
condition_two=(merged["Holiday"].notnull())
condition=[condition_one,condition_two]
choices=["False","True"]
merged['holidays']=np.select(condition,choices)


#Splitting columns into three quantiles
merged["AirIndicator"]=pd.cut(merged["AirTemperature"],bins=3, labels=["low", "medium", "high"], include_lowest=True)
merged["WindIndicator"]=pd.cut(merged["Windspeed"],bins=3, labels=["low", "medium", "high"], include_lowest=True)

merged.to_csv("exploratory.csv",sep=",",encoding='utf-8',index=False)

#Check dates
#print(merged.groupby([merged["Date"].dt.date])["Date"].count())


top_check_ins = pd.DataFrame(merged.groupby(merged["Address"])["Check In"].sum().sort_values(ascending=False).head(10))
top_check_ins = pd.merge(top_check_ins, merged, on="Address")
#print("Top 10 check in stations:")
#print(top_check_ins)

top_check_outs = pd.DataFrame(merged.groupby(merged["Address"])["Check Out"].sum().sort_values(ascending=False).head(10))
top_check_outs = pd.merge(top_check_outs,merged, on="Address")
#print("Top 10 check out stations:")
#print(top_check_outs)

merged["Total Activity"] = merged["Check In"] + merged["Check Out"]
merged1=merged.copy()
merged1 =merged1.groupby(merged1["Address"])["Total Activity"].sum()


top_activity = merged1.copy().sort_values(ascending=False).head(10)
print("Top 10 busiest stations:")
print(top_activity)

low_activity = merged1.copy().sort_values().head(10)
print("Top 10 Non-busy stations:")
print(low_activity)


##############################################################
################# FIND AVERAGE USAGE PER DAY #################
##############################################################

Help.create_folder(Help.PLOTS_DIR)

average_checkins_per_day = merged.copy()
average_checkins_per_day = average_checkins_per_day.groupby(["Number", "Name", "Weekday"])["Check In"].mean()
average_checkins_per_day = average_checkins_per_day.unstack()
average_checkins_per_day.boxplot(Help.SHORT_WEEKDAY_ORDER)
plt.title("")   
plt.suptitle("")    # get rid of the default title of box plotting
plt.ylabel("Average Checkins")
plt.savefig(Help.PLOTS_DIR + "/Boxplot_Average_checkins_day.png")
plt.gcf().clear()


average_checkouts_per_day = merged.copy()
average_checkouts_per_day  = average_checkouts_per_day .groupby(["Number", "Name", "Weekday"])["Check Out"].mean()
average_checkouts_per_day  =average_checkouts_per_day .unstack()
average_checkouts_per_day.boxplot(Help.SHORT_WEEKDAY_ORDER)
plt.title("")   
plt.suptitle("")   
plt.ylabel("Average Checkouts")
plt.savefig(Help.PLOTS_DIR + "/Boxplot_Average_checkouts_day.png")
plt.gcf().clear()


usage_per_day =merged.copy()
usage_per_day=usage_per_day.groupby(["Number", "Name", "Weekday"])["Total Activity"].sum()
usage_per_day=usage_per_day.unstack()
usage_per_day.boxplot(Help.SHORT_WEEKDAY_ORDER)
plt.title("")   
plt.suptitle("")    
plt.ylabel("Usage Per Day")
plt.savefig(Help.PLOTS_DIR + "/Boxplot_Usage_per_day.png")
plt.gcf().clear()

#Usage per hour at a high activity station
usage_hour =merged.copy()
usage_hour=usage_hour[usage_hour["Number"]==5].groupby(["Time"])["Total Activity"].mean()
usage_hour.plot(x="Time",rot=0,kind='bar',color="darkviolet")
plt.gcf().autofmt_xdate(rotation = "30")
plt.xlabel("Time") 
plt.ylabel("Mean of Total Activity") 
plt.title("Usage per hour at Charlemont Street") 
plt.savefig(Help.PLOTS_DIR + "/Barplot_Usage_hour_at_Charlemont street.png")
plt.gcf().clear()

#Usage per hour at low activity station
usage_hour_low =merged.copy()
usage_hour_low=usage_hour_low[usage_hour_low["Number"]==81].groupby(["Time"])["Total Activity"].mean()
usage_hour_low.plot(x="Time",rot=0,kind='bar',color="gray")
plt.gcf().autofmt_xdate(rotation = "30")
plt.xlabel("Time") 
plt.ylabel("Mean of Total Activity") 
plt.title("Usage per hour at  St James Hospital") 
plt.savefig(Help.PLOTS_DIR + "/Barplot_Usage_hour_at_JamesHosp.png")
plt.gcf().clear()

#Usage during Holidays
usage_hour_holiday=merged.copy()
usage_hour_holiday=usage_hour_holiday[usage_hour_holiday["holidays"]=="True"].groupby(["Time"])["Total Activity"].mean()
usage_hour_holiday.plot(x="Time",rot=0,kind='bar',color="coral")
plt.gcf().autofmt_xdate(rotation = "30")
plt.xlabel("Time") 
plt.ylabel("Mean of Total Activity") 
plt.title(" Holiday mean activity")
plt.savefig(Help.PLOTS_DIR + "/Barplot_Usage_holiday.png")
plt.gcf().clear() 

#Usage during non holidays
usage_nonholiday=merged.copy()
usage_nonholiday=usage_nonholiday[usage_nonholiday["holidays"]=="False"].groupby(["Time"])["Total Activity"].mean()
usage_nonholiday.plot(x="Time",rot=0,kind='bar',color="coral")
plt.gcf().autofmt_xdate(rotation = "30")
plt.xlabel("Time") 
plt.ylabel("Mean of Total Activity") 
plt.title(" Non-Holiday mean activity") 
plt.savefig(Help.PLOTS_DIR + "/Barplot_Usage_not_holiday.png")
plt.gcf().clear() 

#Usage on the basis of Air temperature
usage_temperature=merged.copy()
usage_temperature=usage_temperature.groupby(["AirIndicator"])["Total Activity"].mean()
usage_temperature.plot(x="AirIndicator",rot=0,kind='bar',color="forestgreen")
plt.xlabel("AirIndicator") 
plt.ylabel("Mean of Total Activity") 
plt.title("Mean Usage with Temperature") 
plt.savefig(Help.PLOTS_DIR + "/Barplot_airtemperature.png")
plt.gcf().clear() 


#Usage on the basis of Windspeed
usage_windspeed=merged.copy()
usage_windspeed=usage_windspeed.groupby(["WindIndicator"])["Total Activity"].mean()
usage_windspeed.plot(x="WindIndicator",rot=0,kind='bar',color="cyan")
plt.xlabel("WindIndicator") 
plt.ylabel("Mean of Total Activity") 
plt.title("Mean Usage with Windspeed") 
plt.savefig(Help.PLOTS_DIR + "/Barplot_Windspeed.png")
plt.gcf().clear() 

