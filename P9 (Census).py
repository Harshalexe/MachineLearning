import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/admin/Desktop/project1/Census Income/adult.csv")
df.head()

print(df.shape)

df.replace("?", np.nan, inplace=True)

print('Dataset columns with null values:\n', df.isnull().sum())

for i in df.columns:
    print("Column Name:",i, "Count",df[i].nunique())
    print("Unique Values",df[i].unique())
    print("\n")

freq_workclass = df['workclass'].value_counts().idxmax()
freq_occupation = df['occupation'].value_counts().idxmax()
freq_country = df['native.country'].value_counts().idxmax()

col = ['workclass', 'occupation', 'native.country']
val = [freq_workclass, freq_occupation, freq_country]
for i in range(len(col)):
    df[col[i]].fillna(val[i], inplace=True)
print('Dataset columns with null values:\n', df.isnull().sum())

df.info()
numeric_columns = ['fnlwgt', 'capital.gain', 'capital.loss']

for column in numeric_columns:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.histplot(data=df, x=column, kde=True)  # Use kde=True to overlay a kernel density estimate
    plt.title(f'Distribution of {column}')
    plt.show()

df['AgeBin'] = pd.cut(df['age'].astype(int), 5)
df['HourBin'] = pd.cut(df['hours.per.week'].astype(int), 5)

X = df.drop(['income','age','hours.per.week'], axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.39, random_state = 0)

for i in ['capital.gain', 'capital.loss','fnlwgt']:
    X_train[i] = np.log1p(X_train[i])
    X_test[i] = np.log1p(X_test[i])

categoric_columns = ['relationship', 'race', 'sex']

X_train = pd.get_dummies(data = X_train, columns = categoric_columns)
X_test = pd.get_dummies(data = X_test, columns = categoric_columns)

le =  LabelEncoder()

categoric_columns = X_train.select_dtypes(include=['object','category']).columns

for feature in categoric_columns:
        X_train[feature] = le.fit_transform(X_train[feature])
        X_test[feature] = le.transform(X_test[feature])

model = LogisticRegression(max_iter=1000, solver = 'saga')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Logistic Regression accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


