
# coding: utf-8

# In[208]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[209]:

filename = 'train.csv'
#filename = 'test.csv'
df = pd.read_csv(filename)
df.head()


# In[210]:

df.columns = ['Id', 'ints']
a = df.ix[0].ints.split(',')[-1]


# In[211]:

df['ints_list'] = df.ints.apply(lambda x: x.split(','))
df['ints_len'] = df.ints_list.apply(lambda x: len(x))
df['ints_data'] = df.ints_list.apply(lambda x: x[:-1])
df['res'] = df.ints_list.apply(lambda x: x[-1])
df['prev'] = df.ints_list.apply(lambda x: int(x[-2]) if len(x) > 1 else np.nan)
df.drop('ints', axis=1, inplace=True)
print('Longest:', df.ints_len.max())
df.head()


# In[212]:

df.ints_len.max()
nrows, ncols = 4, 4
ar = np.random.randint(low=0, high=len(df.index), size=nrows*ncols)
print('Indices:', ar)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,20))
sns.set(style='whitegrid', context='talk')
#for i in range(nrows * ncols):
cnt = 0
for i in ar:    
    row = df.ix[i]
    x = np.arange(row.ints_len)
    y = row.ints_data
    res = row.res
    plt.subplot(nrows, ncols, cnt + 1)
    plt.scatter(x[:-1], y, c='g', label=str(i))
    plt.scatter(x[-1], res, c='r', s=30)
    plt.legend(loc='upper left')
    cnt += 1
sns.despine()


# In[ ]:

df1 = df.iloc[44349]
df_train = pd.DataFrame()
df_train['x'] = np.arange(df1.ints_len)
df_train['x2'] = df_train.x ** 2
df_train['x3'] = df_train.x ** 3
df_train['x4'] = df_train.x ** 4
df_train['x5'] = df_train.x ** 5
df_train['sqrt'] = df_train.x ** .5
df_train['exp'] = np.exp(df_train.x).astype(float)
df_train['x0'] = 1
df_train['odd'] = df_train.x % 2
df_train['log'] = np.log(df_train.x).astype(float)
df_train['res'] = list(map(int, df1.ints_list))
df_train['prev'] = df_train.res.shift(1)
df_train['diff'] = df_train.res - df_train.prev
#df_train['def'] = default_value
df_train = df_train[['x0', 'x', 'x2', 'x3', 'x4', 'x5', 'sqrt', 'odd', 'exp', 'log', 'sin', 'cos', 'prev', 'res']]
#df_train


# In[216]:

ints_ser = pd.Series(list(map(int, df1.ints_list)))
print('mean:', np.mean(ints_ser))
print('last:', df1.ints_list[-1])
try:
    freqs = ints_ser.value_counts() 
    if len(freqs.index) > 2:
        freq, next_freq = freqs.iloc[0], freqs.iloc[1]
        default_value = freqs.idxmax() if freq / next_freq >= 2 else df1.ints_list[-2]
        print('most frequent value: %d (freq = %d)' % (freqs.idxmax(), freqs.iloc[0]))
        print(freqs.head().to_string())
    else:
        default_value = df1.ints_list[-2]
except OverflowError:
    print('overflow')
    default_value = df1.ints_list[-2]

print('default value = ', default_value)


# In[218]:

# Check recursion

def get_matrix(seq, order):
    A = []
    for i in range(order + 1):
        s = [1] + seq[i:i+order]
        A = s if i == 0 else np.vstack([A, s]) 
    b = seq[order:2*order+1]
    return A, b

def check_recursion(seq):
    p = len(seq) - 1
    n = int(p/2 -1)
    try:
        A, b = get_matrix(seq, n)
        if  np.linalg.matrix_rank(A) - 1 < n:
            n = np.linalg.matrix_rank(A) - 1
            A, b = get_matrix(seq, n)
        w = np.linalg.solve(A, b)    
        # check
        feat_check_A = [1] + seq[n+1:2*n+1]
        feat_check_b = seq[2*n+1]
        if np.dot(feat_check_A, w) == feat_check_b:        
            feat_pred_A = [1] + seq[p-n+1:]
            predict = np.dot(feat_pred_A, w)
            return int(predict)
        else: return np.nan        
    except:
        return np.nan


# In[220]:

print('Recursion:', check_recursion(list(map(int, df1.ints_data))))


# In[ ]:

cols = [col for col in df_train.columns if col != 'res']
df_train.replace(np.inf, np.nan, inplace=True)
df_train.replace(-np.inf, np.nan, inplace=True)
df_train.dropna(inplace=True)
df_train = df_train.reset_index()
print(cols)


# In[12]:

from sklearn import linear_model
lr = linear_model.LinearRegression()
X_train = df_train[:-1][cols]
y_train = df_train[:-1].res
X_test = df_train[cols]
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
df_train['predict'] = pred


# In[13]:

print(X_train.tail(10).to_string(), '\n')
print(X_test.tail(10).to_string(), '\n')
print(y_train.tail(10).to_string(), '\n')
print(pred[-1])


# In[14]:

from sklearn import cross_validation
import warnings

alphas = [10 ** x for x in range(-5, 5)]
with warnings.catch_warnings():
    warnings.filterwarnings('error')
    try:
        lr_rigdeCV = linear_model.RidgeCV(alphas=alphas)
        lr_rigdeCV.fit(X_train, y_train)
        best_alpha = lr_rigdeCV.alpha_        
    except: best_alpha = 0.1
    
print('Alpha =', best_alpha)    
lr_ridge = linear_model.Ridge(alpha=best_alpha)
lr_ridge.fit(X_train, y_train)
pred_ridge = lr_ridge.predict(X_test)
df_train['predict_ridge'] = pred_ridge

kf = cross_validation.KFold(len(X_train.index), n_folds=5, shuffle=True)
scores = cross_validation.cross_val_score(lr_ridge, X_train, y_train, cv=kf)
print('Cross validation score =', np.mean(scores))


# In[15]:

import seaborn as sns
sns.set(style='whitegrid', context='talk')
plt.scatter(df_train[:-1].x, df_train[:-1].res, s=20, c='g', label='Train set')

for j in range(5):    
    plt.scatter(df_train[-1:].x, df_train[-1:].res, s=50+5**j, c='green', alpha=0.52-0.12*j, label='_')
    plt.scatter(df_train[-1:].x, df_train[-1:].predict,  s=50+5**j, c='red', alpha=0.52-0.12*j, label='_')
    plt.scatter(df_train[-1:].x, df_train[-1:].predict_ridge,  s=50+5**j, c='darkblue', alpha=0.52-0.12*j, label='_')
    
plt.plot(df_train.x, df_train.predict, '-', c='r', lw=0.8, ms=5, label='LinReg')
plt.plot(df_train.x, df_train.predict_ridge, '-', c='darkblue', lw=0.8, ms=5, label='LinReg Ridge')
plt.legend(loc='upper left', frameon=True)
#sns.despine()
res = df_train[-1:].res
lr_p = int(np.round(float(df_train[-1:].predict.values[0])))
lrr_p = int(np.round(float(df_train[-1:].predict_ridge.values[0])))
plt.title('Real value = %s,\nLinReg pred = %s,LinReg Ridge pred = %s,\ndefault_value = %d'
          % (str(df_train[-1:].res.values[0]), str(lr_p), str(lrr_p), int(default_value)))


# In[518]:

st = time.time()
#df['recursion'] = df.ints_data.apply(lambda x: check_recursion(list(map(int, x))))
#print('Recursions checked. Time elapsed: %.2f min' % ((time.time() - st) / 60))
#df[['Id', 'recursion']].to_csv('train_recursion_result.csv', encoding='utf-8-sig')
df['recursion'] = df['Id'].map(pd.read_csv('train_recursion_result.csv', encoding='utf-8-sig').set_index('Id').recursion)
df.head()


# In[ ]:

# check some recursions
import time
st = time.time()
def get_diffs(x):
    return [x[i+1] - x[i] for i in range(len(x) - 1)] if len(x) > 1 else x    

df['ints_data'] = df.ints_data.apply(lambda x: list(map(int, x)))
df['diffs'] = df.ints_data.apply(get_diffs)
df['plus1'] = df.ints_data.apply(lambda x: [i+1 for i in x])
df['minus1'] = df.ints_data.apply(lambda x: [i-1 for i in x])
print('time: %.2f min', ((time.time() - st) / 60))


# In[ ]:

st = time.time()
df['recursion'] = df['Id'].map(pd.read_csv('train_recursion_result.csv', encoding='utf-8-sig').set_index('Id').recursion)
df['rec_diffs'] = df.apply(lambda row: check_recursion(row.diffs) + row.prev, axis=1)
df['rec_plus1'] = df.plus1.apply(lambda x: check_recursion(x) - 1)
df['rec_minus1'] = df.minus1.apply(lambda x: check_recursion(x) + 1)
print('time: %.2f min', ((time.time() - st) / 60))


# In[ ]:

print(df[df.recursion.notnull()].Id.count() / sample_n)
df.recursion.fillna(df.rec_diffs, inplace=True)
df.recursion.fillna(df.rec_plus1, inplace=True)
df.recursion.fillna(df.rec_minus1, inplace=True)
print(df[df.recursion.notnull()].Id.count() / sample_n)
df[['Id', 'recursion']].to_csv('train_full_recursion_result.csv', encoding='utf-8-sig')


# In[533]:

from sklearn import linear_model 
from sklearn import cross_validation
import time
preds = []
longest = df.ints_len.max() + 1
#cnt = 200
#cnt = 10000
cnt = len(df.index)

df_train = pd.DataFrame()
df_train['x'] = np.arange(longest)
df_train['x0'] = 1
df_train['x2'] = df_train.x ** 2
df_train['x3'] = df_train.x ** 3
df_train['x4'] = df_train.x ** 4
df_train['sqrt'] = df_train.x ** .5
df_train['exp'] = np.exp(df_train.x)
df_train['odd'] = df_train.x % 2
df_train['log'] = np.log(df_train.x)
df_train['sin'] = np.sin(df_train.x)
df_train['cos'] = np.cos(df_train.x)

alphas = [10 ** x for x in range(-4, 4)]
st = time.time()
for i in range(cnt):
    if (i % 1000 == 0) & (i != 0):
        print('Sequences: %d of %d, time elapsed: %.2f min, total estimated time: %.2f min' 
              % (i, cnt, (time.time() - st) / 60, (time.time() - st) * cnt / (60 * i)))        
    #print(i)
    df1 = df.ix[i]    
    df_curr = df_train[:df1.ints_len].copy(deep=True)    
    #print(df1.ints_len, len(df_curr.index))
    df_curr['res'] = df1.ints_list
    df_curr['prev'] = df_curr.res.shift(1)
    ints_ser = pd.Series(list(map(int, df1.ints_list)))
    def_func = df1.ints_data[-1] if len(df1.ints_data) > 0 else 1
    #def_func = np.mean(ints_ser)
    try:
        freqs = ints_ser[ints_ser < np.iinfo(np.int64).max].value_counts() 
        if len(freqs.index) > 2:
            freq, next_freq = freqs.iloc[0], freqs.iloc[1]
            default_value = freqs.idxmax() if freq / next_freq >= 2 else def_func
        else:
            default_value = def_func
    except OverflowError:
        default_value = def_func
    
    df_curr['def'] = default_value
    df_curr = df_curr[['x0', 'x', 'x2', 'x3', 'x4', 'sqrt', 'odd', 'exp', 'log', 'sin', 'cos', 'prev', 'def', 'res']]

    cols = [col for col in df_curr.columns if col != 'res']    
    df_curr = df_curr[1:].reset_index()
    
    if not df_curr[:-1].empty:
        X_train = df_curr[:-1][cols]
        y_train = df_curr[:-1].res        
        X_test = df_curr[cols]        
        
        lr = linear_model.LinearRegression()        
        lr.fit(X_train, y_train)        
        pred = lr.predict(X_test)
        df_curr['predict'] = pred        
        #preds.append(df_curr[-1:].predict)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")            
            try:
                lr_rigdeCV = linear_model.RidgeCV(alphas=alphas)
                lr_rigdeCV.fit(X_train, y_train)
                best_alpha = lr_rigdeCV.alpha_
            except: best_alpha = 0.1
            
            try:
                lr_ridge = linear_model.Ridge(alpha=best_alpha)
                lr_ridge.fit(X_train, y_train)
                pred_ridge = lr_ridge.predict(X_test)
            except: pred_ridge = pred
                
            if len(X_train.index) > 5:
                try:
                    kf = cross_validation.KFold(len(X_train.index), n_folds=5, shuffle=True)
                    scores = cross_validation.cross_val_score(lr_ridge, X_train, y_train, cv=kf)
                    col_to_result = 'def' if np.mean(scores) < 0.999 else 'predict_ridge'                    
                    #if np.mean(scores) >= 0.999:
                    #    if np.round(pred_ridge[-1]) == int(df1.ints_list[-1]): 
                    #        print('SUCCESS!!!')
                    #        print('CV score =', np.mean(scores))
                    #        print('Predict', int(np.round(pred_ridge[-1])), ', real', df1.ints_list[-1])
                except: col_to_result = 'predict_ridge'
            else: col_to_result = 'predict_ridge'

        df_curr['predict_ridge'] = pred_ridge
        preds.append(df_curr[-1:][col_to_result].values[0])
        
    else:
        preds.append(default_value)
    
print(len(df.index), len(preds))
print('Time %.2f sec' % (time.time() - st))
print('Estimated time for full test data: %.2f min' % ((time.time() - st) * int(len(df.index)) / (60 * int(len(preds)))))
print(preds[:20])


# In[539]:

print(len(preds))
if 'preds' in df.columns:
    df.drop('preds', axis=1, inplace=True)

df = df.join(pd.Series(preds, name='preds').to_frame())
df[['Id', 'ints_data', 'recursion', 'preds']].head()


# In[541]:

df['Last'] = df.recursion
df.Last.fillna(df.preds, inplace=True)
df.Last.fillna(df.prev, inplace=True)
df.Last.fillna(1, inplace=True)
#df[df.Last.isnull()]
df['Last'] = df['Last'].apply(lambda x: int(np.round(float(x))))
df[['Id', 'ints_data', 'res', 'recursion', 'preds']].head(10)


# In[542]:

# check train
df['res_str'] = df.res.apply(str)
df['last_str'] = df.Last.apply(str)
df[df.res_str == df.last_str].Id.count() / df.Id.count()


# In[ ]:

#df[['Id', 'Last']].to_csv('submission.csv', sep=',', index=False)

