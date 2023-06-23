import pandas as pd
import os
for dirname, filenames in os.walk('C:/nskdl'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, roc_auc_score
import scipy
from scipy import stats
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
import os
import warnings
import gc
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_fscore_support, classification_report


warnings.simplefilter('ignore')

columns = (['duration'
    , 'protocol_type'
    , 'service'
    , 'flag'
    , 'src_bytes'
    , 'dst_bytes'
    , 'land'
    , 'wrong_fragment'
    , 'urgent'
    , 'hot'
    , 'num_failed_logins'
    , 'logged_in'
    , 'num_compromised'
    , 'root_shell'
    , 'su_attempted'
    , 'num_root'
    , 'num_file_creations'
    , 'num_shells'
    , 'num_access_files'
    , 'num_outbound_cmds'
    , 'is_host_login'
    , 'is_guest_login'
    , 'count'
    , 'srv_count'
    , 'serror_rate'
    , 'srv_serror_rate'
    , 'rerror_rate'
    , 'srv_rerror_rate'
    , 'same_srv_rate'
    , 'diff_srv_rate'
    , 'srv_diff_host_rate'
    , 'dst_host_count'
    , 'dst_host_srv_count'
    , 'dst_host_same_srv_rate'
    , 'dst_host_diff_srv_rate'
    , 'dst_host_same_src_port_rate'
    , 'dst_host_srv_diff_host_rate'
    , 'dst_host_serror_rate'
    , 'dst_host_srv_serror_rate'
    , 'dst_host_rerror_rate'
    , 'dst_host_srv_rerror_rate'
    , 'attack'
    , 'level'])
df_train = pd.read_csv(r"C:/Users/admin/Desktop/project1/10th/archive (1)/KDDTrain+.txt", header=None, names=columns)
df_test = pd.read_csv(r"C:/Users/admin/Desktop/project1/10th/archive (1)/KDDTest+.txt", header=None, names=columns)

print(df_train)

print(df_train['attack'].unique())

print(df_train.hist(figsize=(16, 12)))

print(df_train.info())

df_train['attack'].describe()
print(df_train['attack'].value_counts())

print(df_train.duplicated().sum())
print(df_test.duplicated().sum())

print(df_train.isnull().sum())

print(df_train['attack'].unique())

print(df_train.head())

cat_features = [i for i in df_train.columns if df_train.dtypes[i] == 'object']

print(cat_features)

print(df_train['protocol_type'].unique())

print(df_train['service'].value_counts())

print(df_train['flag'].unique())

print(df_train['attack'].value_counts())

df_train["binary_attack"] = df_train.attack.map(lambda a: "normal" if a == 'normal' else "abnormal")
df_train.drop('attack', axis=1, inplace=True)

df_test["binary_attack"] = df_test.attack.map(lambda a: "normal" if a == 'normal' else "abnormal")
df_test.drop('attack', axis=1, inplace=True)

print(df_train['binary_attack'].value_counts())

print(df_train.select_dtypes(['object']).columns)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
clm = ['protocol_type', 'service', 'flag', 'binary_attack']
for x in clm:
    df_train[x] = le.fit_transform(df_train[x])
    df_test[x] = le.fit_transform(df_test[x])

print(df_train['service'].value_counts())

corr = df_train.corr()

plt.figure(figsize=(15, 12))

sns.heatmap(corr)

plt.show()

df_train.drop('num_root', axis=1, inplace=True)


df_train.drop('srv_serror_rate', axis=1, inplace=True)

df_train.drop('srv_rerror_rate', axis=1, inplace=True)

df_train.drop('dst_host_srv_serror_rate', axis=1, inplace=True)

df_train.drop('dst_host_serror_rate', axis=1, inplace=True)

df_train.drop('dst_host_rerror_rate', axis=1, inplace=True)

df_test.drop('num_root', axis=1, inplace=True)

df_test.drop('srv_serror_rate', axis=1, inplace=True)

df_test.drop('srv_rerror_rate', axis=1, inplace=True)

df_test.drop('dst_host_srv_serror_rate', axis=1, inplace=True)

df_test.drop('dst_host_serror_rate', axis=1, inplace=True)

df_test.drop('dst_host_rerror_rate', axis=1, inplace=True)


x_train = df_train.drop('binary_attack', axis=1)
y_train = df_train["binary_attack"]

x_test = df_test.drop('binary_attack', axis=1)
y_test = df_test["binary_attack"]

binary_model = RandomForestClassifier()
binary_model.fit(x_train, y_train)
binary_predictions = binary_model.predict(x_test)

base_rf_score = accuracy_score(binary_predictions, y_test)
print(base_rf_score)

x_train['flag'].value_counts().plot(kind='bar')

print(x_train)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

binary_model = RandomForestClassifier()
binary_model.fit(x_train, y_train)
binary_predictions = binary_model.predict(x_test)

base_rf_score = accuracy_score(binary_predictions, y_test)
print(base_rf_score)

print(x_train)

models = {}

from sklearn.linear_model import LogisticRegression

models['Logistic Regression'] = LogisticRegression(multi_class='multinomial')

# Support Vector Machines
from sklearn.svm import LinearSVC

models['Support Vector Machines linear'] = LinearSVC()
models['Support Vector Machines plonomial'] = SVC(kernel='poly')
models['Support Vector Machines RBf'] = SVC(C=100.0)

# Decision Trees
from sklearn.tree import DecisionTreeClassifier

models['Decision Trees'] = DecisionTreeClassifier(max_depth=3)

# Random Forest
from sklearn.ensemble import RandomForestClassifier

models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

models['K-Nearest Neighbor'] = KNeighborsClassifier(n_neighbors=20)

from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy, precision, recall = {}, {}, {}

for key in models.keys():
   
    models[key].fit(x_train, y_train)

    predictions = models[key].predict(x_test)

    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()

print(df_model)


