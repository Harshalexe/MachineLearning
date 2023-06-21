import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


df=pd.read_csv("C:/Users/admin/Desktop/project1/HousingData.csv")
# print(df.isnull().sum())

# deal with Nan
df['CRIM'].fillna((df['CRIM'].mean()), inplace=True)
df['ZN'].fillna((df['ZN'].mean()), inplace=True)
df['INDUS'].fillna((df['INDUS'].mean()), inplace=True)
df['CHAS'].fillna((df['CHAS'].mean()), inplace=True)
df['AGE'].fillna((df['AGE'].mean()), inplace=True)
df['LSTAT'].fillna((df['LSTAT'].mean()), inplace=True)
# print(df.isnull().sum())


le=LabelEncoder()
le.fit(df["MEDV"])
df['MEDV']=le.transform(df['MEDV'])
print(df)
df=df.drop("PTRATIO",axis=1)
df=df.drop("INDUS",axis=1)
df=df.drop("NOX",axis=1)
df=df.drop("TAX",axis=1)
df=df.drop("B",axis=1)
x=df.drop("MEDV",axis=1)
y=df["MEDV"]

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier() #use to select important features
model.fit(x,y) #Take data and put in function/module
print("Importances:",model.feature_importances_) # important features are selected
feat_importance=pd.Series(model.feature_importances_,index=x.columns)
feat_importance.nlargest(10).plot(kind='barh')
plt.show()


from collections import Counter
# print("Count:",Counter(y))
sms = RandomOverSampler(random_state=0)
x,y = sms.fit_resample(x,y)
# print(Counter(y))

from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['DIS']) # for LSTAT, RM, CRIM, DIS
plt.show()
a=['DIS','LSTAT','RM','CRIM']
for i in a:
    print(x[i])
    Q1=x[i].quantile(0.25)
    Q3=x[i].quantile(0.75)
    IQR=Q3-Q1
    print("IQR:",IQR)
    upper=Q3+1.5*IQR
    lower=Q1-1.5*IQR
    print(upper)
    print(lower)
    out1=x[x[i]<lower].values
    out2=x[x[i]>upper].values
    x[i].replace(out1,lower,inplace=True)
    x[i].replace(out2,upper,inplace=True)
    sns.boxplot(x[i])
    plt.show()


rfc=RandomForestClassifier()
dtc=DecisionTreeClassifier()

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0,test_size=0.3)

func=[rfc,dtc]


for item in func:
    item.fit(xtrain,ytrain)
    ypred=item.predict(xtest)
    h=(accuracy_score(ytest,ypred))
    print("mean square:",mean_squared_error(ytest,ypred))
    print(item,h*100,"%")
