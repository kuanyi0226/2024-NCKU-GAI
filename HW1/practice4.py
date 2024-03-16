import numpy as np
import pandas as pd

#practice4: 條件篩選

# 讀取資料
# 資料中第一列（row）為中文欄位名稱，第二列為英文欄位名稱
# 我們選擇使用英文欄位名稱進行操作
weather_data = pd.read_csv('./data/467410-2022-08.csv', skiprows=1)

    