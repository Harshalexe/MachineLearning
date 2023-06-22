import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
test=pd.read_csv("C:/Users/admin/Desktop/project1/Black Friday Sales/test.csv")
train=pd.read_csv("C:/Users/admin/Desktop/project1/Black Friday Sales/train.csv")
# print(train.isnull().sum())
# print(test.isnull().sum())

df=train._append(test)
# print(df)

le=LabelEncoder()
df['Age']=le.fit_transform(df['Age'])
df['Gender']=le.fit_transform(df['Gender'])
df['City_Category']=le.fit_transform(df['City_Category'])

df['Product_Category_2'].fillna((df['Product_Category_2'].mean()), inplace=True)
df['Product_Category_3'].fillna((df['Product_Category_3'].mean()), inplace=True)

# print(df.head())
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+','')
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)

# from matplotlib import pyplot as plt
# import seaborn as sns
# sns.barplot(x='Age',y='Purchase',hue='Gender',data=df)
# plt.show()
# sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)
# plt.show()
# sns.barplot(x='Product_Category_1',y='Purchase',hue='Gender',data=df)
# plt.show()
# sns.barplot(x='Product_Category_2',y='Purchase',hue='Gender',data=df)
# plt.show()
# sns.barplot(x='Product_Category_3',y='Purchase',hue='Gender',data=df)
# plt.show()

test=df[df['Purchase'].isnull()]
train=df[~df['Purchase'].isnull()]
x=train.drop('Purchase',axis=1)
y=train['Purchase']

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
lr=GradientBoostingRegressor()
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0,test_size=0.3)

xtrain.drop('Product_ID',axis=1,inplace=True)
xtest.drop('Product_ID',axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.transform(xtest)
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)
print(r2_score(ytest,ypred))

