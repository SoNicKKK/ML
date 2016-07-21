
# coding: utf-8

# In[76]:

import numpy as np
import pandas as pd


# In[77]:

df = pd.read_csv('./input/train.csv')
df_init = df.copy(deep=True)
#df = pd.read_csv('./input/test.csv')


# In[78]:

data_cols = [col for col in df.columns if col != 'label']
df_data = df[data_cols]
df_target = df.label


# In[79]:

print(len(df_data.columns))
drop_list = []
for col in df_data.columns:
    l = df_data[col].unique()
    if len(l) == 1:
        drop_list.append(col)
        
df_data = df_data.drop(drop_list, axis=1)
print(len(df_data.columns))


# In[80]:

# from sklearn.decomposition import PCA
# pca = PCA(n_components=100)
# pca.fit(df_data)
# print(pca.explained_variance_ratio_.sum())
# df_pca = pca.transform(df_data)


# In[81]:

# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler().fit(df_data)
# df_scaled = scaler.transform(df_data)


# In[82]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def rf(df_):
    clf = RandomForestClassifier(n_estimators=60)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_, df_target, test_size=0.4)
    clf.fit(X_train, y_train)    
    return clf, clf.score(X_test, y_test)

_, sc = rf(df_data)
print('no scale', sc)
_, sc_scale = rf(df_data / 255)
print('with scale', sc_scale)


# In[95]:

df_data_train = df_data.ix[:5000]
df_target_train = df_target.ix[:5000]
from sklearn.ensemble import GradientBoostingClassifier
lr = 0.05
n_est = 50

for n in np.arange(30, 100, 10):
    clf = GradientBoostingClassifier(learning_rate=lr, n_estimators=n)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_data_train, df_target_train, test_size=0.33)
    clf.fit(X_train, y_train)
    print(n, clf.score(X_test, y_test))


# Selected values: learning_rate = 0.05, n_estimators = 80

# In[ ]:

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

n_trees = np.arange(30, 70, 10)
for n in n_trees:
    clf = RandomForestClassifier(n_estimators=n)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_data, df_target, test_size=0.4)
    clf.fit(X_train, y_train)    
    print(n, clf.score(X_test, y_test))


# In[ ]:

from sklearn import neighbors
from sklearn import cross_validation 
clf = neighbors.KNeighborsClassifier(n_neighbors=5)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df_data, df_target, test_size=0.4)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))


# In[ ]:

clf = RandomForestClassifier(n_estimators=60)
clf.fit(df_data, df_target)


# In[ ]:

df_test = pd.read_csv('./input/test.csv')
res = clf.predict(df_test)
print(res)


# In[ ]:

df_test['ImageId'] = np.arange(1, 28001)
df_test['Label'] = res
df_test.head()
df_test[['ImageId', 'Label']].to_csv('./submission/submission.csv', index=False)

