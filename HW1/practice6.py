import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#practice6: 雷達圖

# 分析風速和風向之間的關係，對於每個風向角度區間（0-90度、90-180度、180-270度、270-360度），計算相應的平均風速，並繪製成雷達圖以可視化四種風向的風速分佈情況。

weather_data = pd.read_csv('./data/467410-2022-08.csv', skiprows=1)

weather_data['WD'] = pd.cut(weather_data['WD'], bins=[0, 90, 180, 270, 360], labels=['0-90\ndegree', '90-180\ndegree', '180-270\ndegree', '270-360\ndegree'], right=False)

# Calculate average wind speed for each direction
speedAvg = weather_data.groupby('WD')['WS'].mean()



labels = speedAvg.index.tolist()
values = speedAvg.values.tolist()
ranges = [0,1,2,3,4,5]

# angle for each axis (the angle unit is "radian")
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# complete the radar "circle", "closing" the plot
values = np.concatenate((values,[values[0]]))
angles = np.concatenate((angles,[angles[0]]))

# Plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, polar=True)
ax.plot(angles, values, 'o-', linewidth=2)
ax.fill(angles, values, alpha=0.25)
ax.set_ylim(top=5)

# Add labels
ax.set_yticklabels([])
ax.set_thetagrids(np.array(angles[:-1]) * 180/np.pi, labels)
for range in ranges:
    x = 45
    y = range
    ax.text(x, y, range, ha='center', va='center', fontsize=8, color='gray')

# Add value
for angle, value in zip(angles, values[:-1]):
    x = angle
    y = value - 0.5
    ax.text(x, y, f'{value:.2f}', ha='center', va='center', fontsize=10, color='black')

ax.grid(True)
plt.title('Wind Direction & Speed(m/s) Radar Graph in Aug, 2022')
plt.show()