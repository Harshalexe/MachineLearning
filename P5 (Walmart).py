import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

features = pd.read_csv("C:/Users/admin/Desktop/project1/walmart-recruiting-store-sales-forecasting/features.csv/features.csv")
test = pd.read_csv("C:/Users/admin/Desktop/project1/walmart-recruiting-store-sales-forecasting/test.csv/test.csv")
stores = pd.read_csv("C:/Users/admin/Desktop/project1/walmart-recruiting-store-sales-forecasting/stores.csv")
train = pd.read_csv("C:/Users/admin/Desktop/project1/walmart-recruiting-store-sales-forecasting/train.csv/train.csv")

df1 = train.merge(features, on = ['Store', 'Date', 'IsHoliday'], how = 'inner') #[In this only common column comes]
df = df1.merge(stores, on = ['Store'], how = 'inner')
# print(df.head())

#also you can do like this:
# df2 = stores.merge(features, on = ['Store'], how = 'inner')
# df = df2.merge(train, on = ['Store', 'Date', 'IsHoliday'], how = 'inner')
# print(df.head())

df.drop(['MarkDown1','MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis = 1, inplace=True)
# print(df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date']) #Convert into datetime datatype 
df.set_index(keys = "Date", inplace = True) #To set function in date column
# print(df.head(10))

from matplotlib import pyplot as plt
import seaborn as sns
a=['Temperature', 'Size']
for i in a:
    # print(x[i])
    Q1=df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    IQR=Q3-Q1
    # print("IQR:",IQR)
    upper=Q3+1.5*IQR
    lower=Q1-1.5*IQR
    # print(upper)
    # print(lower)
    out1=df[df[i]<lower].values
    out2=df[df[i]>upper].values
    df[i].replace(out1,lower,inplace=True)
    df[i].replace(out2,upper,inplace=True)
    # sns.boxplot(df[i])
    # plt.show()

#Test

df_test = test.merge(features, on = ['Store', 'Date', 'IsHoliday'], how = 'inner')
test = df_test.merge(stores, on = ['Store'], how = 'inner')

test.drop(["MarkDown1", "MarkDown2","MarkDown3","MarkDown4", "MarkDown5"],axis=1, inplace = True)
# print(test.isnull().sum())

test['CPI'] = test['CPI'].fillna(test['CPI'].mean())
test['Unemployment'] = test['Unemployment'].fillna(test['Unemployment'].mean())

b=['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
for i in b:
    # print(x[i])
    Q1=test[i].quantile(0.25)
    Q3=test[i].quantile(0.75)
    IQR=Q3-Q1
    # print("IQR:",IQR)
    upper=Q3+1.5*IQR
    lower=Q1-1.5*IQR
    # print(upper)
    # print(lower)
    out1=test[test[i]<lower].values
    out2=test[test[i]>upper].values
    test[i].replace(out1,lower,inplace=True)
    test[i].replace(out2,upper,inplace=True)
    # sns.boxplot(test[i])
    # plt.show()

test['Date'] = pd.to_datetime(test['Date'])
test.set_index(keys = 'Date', inplace = True)
df_test.head()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['IsHoliday'] = le.fit_transform(df['IsHoliday'])
df['Type'] = le.fit_transform(df['Type'])
test['IsHoliday'] = le.fit_transform(test['IsHoliday'])
test['Type'] = le.fit_transform(test['Type'])

df.drop(['Size'], axis = 1, inplace = True)

x = df.drop(['Weekly_Sales'], axis = 1)
y = df['Weekly_Sales']

print(y.head())

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=10,test_size=0.3)
from sklearn.linear_model import LinearRegression
lrmodel= LinearRegression()
lrmodel.fit(xtrain,ytrain)
LinearRegression()
prediction1=lrmodel.predict(xtest)
print(lrmodel.score(xtest,ytest))

mse = mean_squared_error(prediction1,ytest)
print("mean sqaure error:",mse)

from sklearn.metrics import r2_score
print(r2_score(prediction1,ytest))

import numpy as np
user_input=np.array([[1,1,0,42.21,2.572,211.096,8.106,0]])
prediction=lrmodel.predict(user_input)
print(prediction)
