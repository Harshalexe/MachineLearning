import pandas as pd

df = pd.read_csv("C:/Users/Harshal/Desktop/ML Intern/uci/TidyData.csv")
print(df)
print(df.isnull().sum())
X = df.drop('Subject', axis=1)
X = X.drop('Activity', axis=1)
Y = df['Activity']

print(X)
print(Y)

from collections import Counter
print(Counter(Y))
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, Y = ros.fit_resample(X, Y)
print(Counter(Y))

from matplotlib import pyplot as plt
import seaborn as sns

out = ['Time domain:Bodyacceleration,standard deviration valueon Z axis', 'Time domain:Gravityacceleration,mean valueon X axis',
       'Time domain:Gravityacceleration,mean valueon Y axis', 'Time domain:Gravityacceleration,mean valueon Z axis',
       'Time domain:Gravityacceleration,standard deviration valueon X axis',
       'Time domain:Gravityacceleration,standard deviration valueon Y axis',
       'Time domain:Gravityacceleration,standard deviration valueon Z axis',
       'Time domain:Bodyacceleration jerk,mean valueon X axis','Time domain:Bodyacceleration jerk,mean valueon Y axis',
       'Time domain:Bodyacceleration jerk,mean valueon Z axis']
for i in out:
    sns.boxplot(df[i])
    plt.show()
    print(X[i])
    Q1 = X[i].quantile(0.25)
    Q3 = X[i].quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)
    upper = Q3 + 1.5 * IQR
    lower = Q1 - 1.5 * IQR
    print(upper)
    print(lower)
    out1 = X[X[i] < lower].values
    out2 = X[X[i] > upper].values
    X[i].replace(out1, lower, inplace=True)
    X[i].replace(out2, upper, inplace=True)
    sns.boxplot(X[i])
    plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier()

X_train,X_test,Y_train,Y_test =  train_test_split(X,Y,random_state=0,test_size=0.3)
rfc.fit(X_train,Y_train)

Y_pred = rfc.predict(X_test)
print(accuracy_score(Y_test,Y_pred)*100)