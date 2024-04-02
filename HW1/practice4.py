import numpy as np
import pandas as pd

#practice4: 條件篩選

# 讀取資料
# 資料中第一列（row）為中文欄位名稱，第二列為英文欄位名稱
# 我們選擇使用英文欄位名稱進行操作
weather_data_original = pd.read_csv('./data/467410-2022-08.csv', skiprows=1)
weather_data = pd.read_csv('./data/467410-2022-08.csv', skiprows=1)

#calculate intensity & mean
weather_data['Precp'] = weather_data['Precp'].replace("T",0).astype(float)
weather_data.loc[weather_data['PrecpHour'] == 0, 'Intensity'] = 0
weather_data.loc[weather_data['PrecpHour'] != 0, 'Intensity'] = weather_data['Precp'] / weather_data['PrecpHour']
precpAvg = weather_data['Intensity'].mean()
print('Intensity Avg: ',precpAvg)

#filter the data
filter = ((weather_data['Precp'] != 'T') & ((weather_data['Intensity'] > precpAvg)))
filtered_data = weather_data[filter]
#filtered_data = weather_data_original.loc[filter, ['Precp',"PrecpHour"]]
print(filtered_data)