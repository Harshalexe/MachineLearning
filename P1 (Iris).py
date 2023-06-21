import pandas as pd
df=pd.read_csv("C:/Users/Harshal/Desktop/ML Intern/iris dataset/Iris.csv")

x=df.drop('Species',axis=1)
y=df['Species'] #selecting one column from dataset
print(x)
print(y)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(score_func=chi2, k='all') 
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_) 
dfcolumns = pd.DataFrame(x.columns)  
featuresScores = pd.concat([dfcolumns,dfscores], axis=1)
featuresScores.columns = ['Specs', 'Score']
print(featuresScores)

print(df.isnull().sum())
df['PetalLengthCm'].fillna((df['PetalLengthCm'].mean()), inplace=True)
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])
print(df)

from imblearn.over_sampling import SMOTE
sms = SMOTE(random_state=0)
x,y = sms.fit_resample(x,y)

from matplotlib import pyplot as plt
import seaborn as sns
sns.boxplot(df['SepalLengthCm'])
plt.show()

print(df['SepalLengthCm'])
Q1=df['SepalLengthCm'].quantile(0.25)
Q3=df['SepalLengthCm'].quantile(0.75)
IQR=Q3-Q1
print("IQR:",IQR)
upper=Q3+1.5*IQR
lower=Q1-1.5*IQR
print(upper)
print(lower)
out1=df[df['SepalLengthCm']<lower].values
out2=df[df['SepalLengthCm']>upper].values
df['SepalLengthCm'].replace(out1,lower,inplace=True)
df['SepalLengthCm'].replace(out2,upper,inplace=True)

from sklearn.linear_model import LogisticRegression 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression() 
pca=PCA(n_components=2)

x=df.drop('Id',axis=1)
x=x.drop('Species',axis=1)
y=df['Species']

pca.fit(x)
x=pca.transform(x)

print(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,random_state=0,test_size=0.3)
logr.fit(xtrain,ytrain) 
ypred=logr.predict(xtest) 
a=(accuracy_score(ytest,ypred))
print(a*100,"%")