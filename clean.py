# -*- coding: utf-8 -*-
"""
Dataset cleaning
"""
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/ERANA/Desktop/DAILY_MERGED.csv',dtype={'Climate_ID': str})
df = df[np.isfinite(df['Total_Precip_mm'])] #Remove empty cells
df = df[np.isfinite(df['MeanTemp'])] #Remove empty cells

#Drop columns with irrelevant data
df = df.drop(['MinTemp','MaxTemp','Date_Time','Total_Rain_mm','Data_Quality','CoolDegDays','Dir_of_Max-Gust_10s_deg','Total_Snow_cm','SpdOfMaxGust_km_h','Max Temp Flag','Min Temp Flag','Mean Temp Flag','Heat Deg Days Flag','Cool Deg Days Flag','Total Rain Flag','Total Snow Flag','Total Precip Flag','Snow_on_Grnd_cm','Snow on Grnd Flag','Dir of Max Gust Flag','Spd of Max Gust Flag','Dir of Max Gust Flag','Spd of Max Gust Flag'], axis=1)

Total_Precip = df['Total_Precip_mm']
outliers = (Total_Precip >= 1)
df[outliers].to_csv('C:/Users/ERANA/Desktop/clean_data.csv', index = False)








