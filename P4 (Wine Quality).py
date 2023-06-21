import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:/Users/admin/Desktop/project1/winequalityN.csv")
print(df.isnull().sum())
df['fixed acidity'].fillna((df['fixed acidity'].mean()), inplace=True)
df['volatile acidity'].fillna((df['volatile acidity'].mean()), inplace=True)
df['citric acid'].fillna((df['citric acid'].mean()), inplace=True)
df['residual sugar'].fillna((df['residual sugar'].mean()), inplace=True)
df['chlorides'].fillna((df['chlorides'].mean()), inplace=True)
df['pH'].fillna((df['pH'].mean()), inplace=True)
df['sulphates'].fillna((df['sulphates'].mean()), inplace=True)
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['type']=le.fit_transform(df['type'])
x=df.drop("quality",axis=1)
y=df['quality']

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k='all') # K is selecting features
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_) #calculate the score respect to chi2
dfcolumns = pd.DataFrame(x.columns)  #Creating columns
featuresScores = pd.concat([dfcolumns,dfscores], axis=1)
featuresScores.columns = ['Feature', 'Score']
print(featuresScores)

from collections import Counter
# print(Counter(y)) # It is use to count
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x, y = ros.fit_resample(x,y)
# print(Counter(y))

from matplotlib import pyplot as plt
import seaborn as sns
a=['fixed acidity', 'volatile acidity', 'citric acid',
       'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
for i in a:
    # print(x[i])
    Q1=x[i].quantile(0.25)
    Q3=x[i].quantile(0.75)
    IQR=Q3-Q1
    # print("IQR:",IQR)
    upper=Q3+1.5*IQR
    lower=Q1-1.5*IQR
    # print(upper)
    # print(lower)
    out1=x[x[i]<lower].values
    out2=x[x[i]>upper].values
    x[i].replace(out1,lower,inplace=True)
    x[i].replace(out2,upper,inplace=True)
    # sns.boxplot(x[i])
    # plt.show()

rfc=RandomForestClassifier()
dtc=DecisionTreeClassifier()

xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0,test_size=0.3)

func=[rfc,dtc]


for item in func:
    item.fit(xtrain,ytrain)
    ypred=item.predict(xtest)
    h=(accuracy_score(ytest,ypred))
    # print("mean square:",mean_squared_error(ytest,ypred))
    print(item,h*100,"%")