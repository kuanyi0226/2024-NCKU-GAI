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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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
df['FamilyEncode'] = 0
df.loc[(df['Family'] == 3) | (df['Family'] == 4), 'FamilyEncode'] = 1
df.loc[(df['Family'] > 4), 'FamilyEncode'] = 2
df['Alone'] = 0
df.loc[df['Family'] == 1, 'Alone'] = 1
# df.loc[df['Pclass'] != 3, 'Pclass'] = 0
# df.loc[df['Pclass'] == 3, 'Pclass'] = 1

df_x = df[['Age', 'Fare','Sex','Pclass']]        
# 取出訓練資料的答案
df_y = df['Survived']

# 數值型態資料前處理
# 創造 imputer 並設定填補策略
#df_x['Ticket'] = df_x['Ticket'].str.extract('(\d+)', expand=False) #extract the number
#df_x['Ticket'] = df.duplicated('Ticket', keep=False).astype(int)
# concat = pd.concat([df['Ticket'],df_x['Ticket']], axis=1)
# print(concat)

imputer_median = SimpleImputer(strategy='median')    
imputer_most_frequent = SimpleImputer(strategy='most_frequent')  
age = df_x['Age'].to_numpy().reshape(-1, 1)
#embarked = df_x['Embarked'].to_numpy().reshape(-1, 1)

fare_mean = df_x['Fare'].mean()
fare_std = df_x['Fare'].std()
df_x['Fare'] = (df_x['Fare']-fare_mean) / fare_std

# 根據資料學習需要填補的值 & 填補缺失值
imputer_median.fit(age)   
#imputer_most_frequent.fit(embarked)                           

df_x['Age'] = imputer_median.transform(age)   
#df_x['Embarked'] = imputer_most_frequent.transform(embarked)   

age_mean = df_x['Age'].mean()
age_std = df_x['Age'].std()
df_x['Age'] = (df_x['Age']-age_mean) / age_std

# 類別型態資料前處理
# 創造 Label Encoder
le = LabelEncoder()
encode_targets = ['Sex']
for target in encode_targets:
    # 給予每個類別一個數值
    le.fit(df_x[target])
    # 轉換所有類別成為數值
    df_x[target] = le.transform(df_x[target])    

#print(df_x.isnull().sum())
# 分割 train and test sets，random_state 固定為 1012
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, train_size=0.8, random_state=1012)

# 1. Random Forest
Randomforest = RandomForestClassifier(
    random_state=1012,
    n_estimators = 28, #28
    criterion = 'gini', 
    #min_samples_leaf=1,
    min_samples_split=2,
    #max_leaf_nodes=26,
    max_depth=6,#6
    )
Randomforest.fit(train_x, train_y)                         

# 2. K Neighbors
KNeighbor =KNeighborsClassifier(n_neighbors=7) #7
KNeighbor.fit(train_x,train_y)

# 3. Logistic Regression
logic = LogisticRegression(random_state = 1012, max_iter=30) #30
logic.fit(train_x,train_y)

# 4. GaussianNB
gNB = GaussianNB()
gNB.fit(train_x,train_y)

# 5. SVC
svc = SVC(kernel='linear',random_state=1012)
svc.fit(train_x,train_y)

# 計算準確度
pred_train_RF = Randomforest.predict(train_x) 
train_acc_RF = accuracy_score(train_y, pred_train_RF)  
pred_test_RF = Randomforest.predict(test_x) 
test_acc_RF = accuracy_score(test_y, pred_test_RF)   

pred_train_KN = KNeighbor.predict(train_x) 
train_acc_KN = accuracy_score(train_y, pred_train_KN)  
pred_test_KN = KNeighbor.predict(test_x) 
test_acc_KN = accuracy_score(test_y, pred_test_KN)    

pred_train_LO = logic.predict(train_x) 
train_acc_LO = accuracy_score(train_y, pred_train_LO)  
pred_test_LO = logic.predict(test_x) 
test_acc_LO = accuracy_score(test_y, pred_test_LO)  

pred_train_G = gNB.predict(train_x) 
train_acc_G = accuracy_score(train_y, pred_train_G)  
pred_test_G = gNB.predict(test_x) 
test_acc_G = accuracy_score(test_y, pred_test_G)  

pred_train_SVC = gNB.predict(train_x) 
train_acc_SVC = accuracy_score(train_y, pred_train_SVC)  
pred_test_SVC = gNB.predict(test_x) 
test_acc_SVC = accuracy_score(test_y, pred_test_SVC)  

# 輸出準確度
print('(1) Random Forest Classifier')
print('train accuracy: {}'.format(train_acc_RF)) 
print('test accuracy: {}'.format(test_acc_RF))
print('(2) K Neighbors Classifier')
print('train accuracy: {}'.format(train_acc_KN)) 
print('test accuracy: {}'.format(test_acc_KN))
print('(3) Logistic Regression')
print('train accuracy: {}'.format(train_acc_LO)) 
print('test accuracy: {}'.format(test_acc_LO))
print('(4) Gaussian NB')
print('train accuracy: {}'.format(train_acc_G)) 
print('test accuracy: {}'.format(test_acc_G))
print('(5) SVC')
print('train accuracy: {}'.format(train_acc_SVC)) 
print('test accuracy: {}'.format(test_acc_SVC))

# save the model
#pickle.dump(Randomforest, open('./models/Titanic_RandomForest.pickle', 'wb'))