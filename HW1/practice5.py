import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#practice5: 折線圖
# 請依照日期畫出氣溫以及雨量的變化，並以折線圖的方式呈現。

weather_data = pd.read_csv('./data/467410-2022-08.csv', skiprows=1)
weather_data['Precp'] = weather_data['Precp'].replace("T",0).astype(float)
fig, ax = plt.subplots(2,1, figsize=(9,6))

# Plot Temperature
ax[0].plot(weather_data['Temperature'], marker='o')
ax[0].set_xlabel('Date')
ax[0].set_ylabel('Temperature')
ax[0].set_title('Temperature in August, 2022')
ax[0].grid(True)

# Plot Precp
ax[1].plot(weather_data['Precp'], marker='x')
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Precp')
ax[1].set_title('Precipitation in August, 2022')
ax[1].grid(True)

plt.tight_layout()
plt.show()