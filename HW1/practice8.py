import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn

# 練習 Hint

# 匯入填補缺失值的工具
from sklearn.impute import SimpleImputer          
# 匯入 Label Encoder
from sklearn.preprocessing import LabelEncoder    
# 匯入模型
from sklearn.ensemble import RandomForestClassifier
# 匯入準確度計算工具
from sklearn.metrics import accuracy_score     
# 匯入 train_test_split 工具
from sklearn.model_selection import train_test_split   
#正規化
from sklearn.preprocessing import RobustScaler

df = pd.read_csv('./data/train.csv')

all_attr = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']
# check null data
#df.info()
#print(df.isna().sum())
# 取出訓練資料需要分析的資料欄位
df['Family'] = df['SibSp'] + df['Parch'] + 1
df_x = df[['Age', 'Fare','Sex','Pclass','Ticket','Embarked','Family']]        
# 取出訓練資料的答案
df_y = df['Survived']

# 數值型態資料前處理
# 創造 imputer 並設定填補策略
df_x['Ticket'] = df_x['Ticket'].str.extract('(\d+)', expand=False) #extract the number
imputer_median = SimpleImputer(strategy='median')    
imputer_most_frequent = SimpleImputer(strategy='most_frequent')  
age = df_x['Age'].to_numpy().reshape(-1, 1)
embarked = df_x['Embarked'].to_numpy().reshape(-1, 1)
fare = df_x['Fare'].to_numpy().reshape(-1, 1)

# 根據資料學習需要填補的值 & 填補缺失值
imputer_median.fit(age)   
imputer_most_frequent.fit(embarked)                           

df_x['Age'] = imputer_median.transform(age)   
df_x['Embarked'] = imputer_most_frequent.transform(embarked)   



# 類別型態資料前處理
# 創造 Label Encoder
le = LabelEncoder()
encode_targets = ['Sex','Embarked']
for target in encode_targets:
    # 給予每個類別一個數值
    le.fit(df_x[target])
    # 轉換所有類別成為數值
    df_x[target] = le.transform(df_x[target])    

# 分割 train and test sets，random_state 固定為 1012
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, train_size=0.8, random_state=1012)

# 1. Random Forest
Randomforest = RandomForestClassifier(random_state=1012,n_estimators = 15, criterion = 'gini',max_leaf_nodes=50,max_depth= 12)
Randomforest.fit(train_x, train_y)                       
pred_train = Randomforest.predict(train_x)    

# 計算準確度
train_acc = accuracy_score(train_y, pred_train)             

# 輸出準確度
print('(1) Random Forest ')
print('train accuracy: {}'.format(train_acc)) 

# 確認模型是否訓練成功
pred_test = Randomforest.predict(test_x)                   
# 計算準確度
test_acc = accuracy_score(test_y, pred_test)             

# 輸出準確度
print('test accuracy: {}'.format(test_acc)) 

# save the model
pickle.dump(Randomforest, open('./models/Titanic_RandomForest.pickle', 'wb'))