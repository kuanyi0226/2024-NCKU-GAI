import numpy as np
import pandas as pd

#practice3: 數值轉換
#https://codis.cwa.gov.tw/StationData Tainan(467410) 2022.08
counter = [0,0,0,0,0]
def UV_level(x):
    if x >= 11:
        counter[4] += 1
        return '極高'
    elif x >= 8:
        counter[3] += 1
        return '甚高'
    elif x >= 6:
        counter[2] += 1
        return '高'
    elif x >= 3:
        counter[1] += 1
        return '中'
    else:
        counter[0] += 1
        return '低'
    
# 讀取資料
# 資料中第一列（row）為中文欄位名稱，第二列為英文欄位名稱
# 我們選擇使用英文欄位名稱進行操作
weather_data = pd.read_csv('./data/467410-2022-08.csv', skiprows=1, 
                           usecols=["ObsTime","UVI Max"]) #skip one row(Chinese title)
#add new column
weather_data_UVlevel = weather_data.assign( UVI_Level=weather_data['UVI Max'].apply(UV_level))
print(weather_data_UVlevel)
#print the counter
print("極高: ",counter[4])
print("甚高: ",counter[3])
print("高: ",counter[2])
print("中: ",counter[1])
print("低: ",counter[0])
    