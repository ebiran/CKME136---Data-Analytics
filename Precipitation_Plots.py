%load Precipitation_Plots.py
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import linear_model
from subprocess import check_output

df = pd.read_csv('C:/Users/ERANA/Desktop/clean_data.csv')
df.info()

stations = df['Name'].unique()
num_of_stations = stations.size
print('Total # of stations: ' + str(num_of_stations))
      
      
df1 = df.groupby(['Name','Year'])['Total_Precip_mm'].sum()
df1 = pd.DataFrame(df1)

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
df1.groupby('Name').mean().sort_values(by='Total_Precip_mm', ascending=False)['Total_Precip_mm'].plot('bar', color='r',width=0.3,title='Average AnnualPrecipitation', fontsize=20)
plt.xticks(rotation = 90)
plt.ylabel('Average Annual Percipitation (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


fig = plt.figure(figsize=(16,10))
ax = fig.add_subplot(111)
dfg = df.groupby('Year')['Total_Precip_mm'].sum()
dfg.plot('line', title='Overall Annual Percipitation', fontsize=20)
plt.ylabel('Overall Percipitation (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
print('Max: ' + str(dfg.max()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.max()].index.values[0:]))
print('Min: ' + str(dfg.min()) + ' ocurred in ' + str(dfg.loc[dfg == dfg.min()].index.values[0:]))
print('Mean: ' + str(dfg.mean()))


#Precipitation by Month

dfm = df.groupby('Month')['Total_Precip_mm'].sum()
dfm = pd.DataFrame(dfm)

fig = plt.figure(figsize=(16,8))
ax = fig.add_subplot(111)
dfm.groupby('Month').mean().sort_values(by='Total_Precip_mm', ascending=False)['Total_Precip_mm'].plot('bar', color='r',width=0.3,title='Average MAnual Precipitation', fontsize=20)
plt.xticks(rotation = 90)
plt.ylabel('Average Monthly Percipitation (mm)')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)


