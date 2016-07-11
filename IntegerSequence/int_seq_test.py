
# coding: utf-8

# In[50]:

import numpy as np
import pandas as pd
import time
import warnings
from sklearn import linear_model 
from sklearn import cross_validation


# In[51]:

filename = 'test.csv'
df = pd.read_csv(filename)
df.columns = ['Id', 'ints']
df['ints_list'] = df.ints.apply(lambda x: x.split(','))
df['ints_len'] = df.ints_list.apply(lambda x: len(x))
df['last'] = df.ints_list.apply(lambda x: int(x[-1]))
df.drop('ints', axis=1, inplace=True)
print('Longest:', df.ints_len.max())
longest = df.ints_len.max() + 1
df.head()


# In[52]:

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
df_train.tail()


# In[53]:

cols = [col for col in df_train.columns if col != 'res']
cols


# In[54]:

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


# In[55]:

# check some recursions
import time
st = time.time()
def get_diffs(x):
    return [x[i+1] - x[i] for i in range(len(x) - 1)] if len(x) > 1 else x    

df['ints_list'] = df.ints_list.apply(lambda x: list(map(int, x)))
df['diffs'] = df.ints_list.apply(get_diffs)
df['plus1'] = df.ints_list.apply(lambda x: [i+1 for i in x])
df['minus1'] = df.ints_list.apply(lambda x: [i-1 for i in x])
print('time: %.2f min' % ((time.time() - st) / 60))


# In[56]:

st = time.time()
#print(df.ix[0])
df['recursion'] = df['Id'].map(pd.read_csv('test_recursion_result.csv', encoding='utf-8-sig').set_index('Id').recursion)
df['rec_diffs'] = df.apply(lambda row: check_recursion(row.diffs) + row['last'], axis=1)
df['rec_plus1'] = df.plus1.apply(lambda x: check_recursion(x) - 1)
df['rec_minus1'] = df.minus1.apply(lambda x: check_recursion(x) + 1)
print('time: %.2f min' % ((time.time() - st) / 60))


# In[57]:

print(df[df.recursion.notnull()].Id.count() / len(df.index))
df.recursion.fillna(df.rec_diffs, inplace=True)
df.recursion.fillna(df.rec_plus1, inplace=True)
df.recursion.fillna(df.rec_minus1, inplace=True)
print(df[df.recursion.notnull()].Id.count() / len(df.index))
df[['Id', 'recursion']].to_csv('test_full_recursion_result.csv', encoding='utf-8-sig')


# In[59]:

st = time.time()

# check recursion and write to file

#df['recursion'] = df.ints_list.apply(lambda x: check_recursion(list(map(int, x))))
#print('Recursions checked. Time elapsed: %.2f min' % ((time.time() - st) / 60))
#print('Recursion detected for %.4f sequences' % (df[df.recursion.isnull() == False].Id.count() / df.Id.count()))
#df[['Id', 'recursion']].to_csv('test_recursion_result.csv', index=False)
#pd.read_csv('test_recursion_result.csv', encoding='utf-8-sig')

# get recursion result from file
df['recursion'] = df['Id'].map(pd.read_csv('test_full_recursion_result.csv', encoding='utf-8-sig').set_index('Id').recursion)
df.head()


# In[61]:

preds = []
longest = df.ints_len.max() + 1
alphas = [10 ** x for x in range(-4, 4)]

#cnt = 5
cnt = len(df.index)

#nrows, ncols = 4, 4
#cnt = nrows * ncols
#fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18,18))
#sns.set(style='whitegrid', context='notebook')

st = time.time()
for i in range(cnt):
    if (i % 2000 == 0) & (i != 0):
        print('Sequences: %d of %d (%.2f%% done), time elapsed: %.2f min, estimated time: %.2f min' 
              % (i, cnt, 100 * i / cnt, (time.time() - st) / 60, (time.time() - st) * cnt / (60 * i)))        
    #print(i)
    df1 = df.ix[i]
    if np.isnan(df1.recursion): # do linear regression only if no recursion detected
        df_curr = df_train[:df1.ints_len + 1].copy(deep=True)       
        df_curr['res'] = df1.ints_list + [np.nan]
        df_curr['prev'] = df_curr.res.shift(1)
        ints_ser = pd.Series(list(map(int, df1.ints_list)))
        default_value = df1.ints_list[-1]
        #def_func = np.mean(ints_ser)
        try:
            freqs = ints_ser[ints_ser < np.iinfo(np.int64).max].value_counts() 
            if len(freqs.index) > 2:
                freq, next_freq = freqs.iloc[0], freqs.iloc[1]
                default_value = freqs.idxmax() if freq / next_freq >= 2 else default_value            
        except OverflowError: default_value

        cols = [col for col in df_curr.columns if col != 'res']    
        df_curr = df_curr[1:].reset_index()

        if len(df_curr.index) > 1:
            X_train = df_curr[:-1][cols]
            y_train = df_curr[:-1].res          
            X_test = df_curr[-1:][cols]

            #lr = linear_model.LinearRegression()
            #lr.fit(X_train, y_train)
            #predict = lr.predict(X_test)[0]  
            #print('predict after LR:', predict)  

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
                    predict = lr_ridge.predict(X_test)[0]
                except: predict = default_value
                #print('predict after ridge LR:', predict)

                if len(X_train.index) > 5:
                    try:
                        kf = cross_validation.KFold(len(X_train.index), n_folds=5, shuffle=True)
                        scores = cross_validation.cross_val_score(lr_ridge, X_train, y_train, cv=kf)
                        predict = default_value if np.mean(scores) < 0.999 else default_value
                    except: predict = default_value
                else: predict = default_value
                #print('predict after cross validation:', predict)

            preds.append(predict)
        else:
            preds.append(default_value)
    else:
        preds.append(df1.recursion)
    
    #plt.subplot(nrows, ncols, i + 1)
    #plt.scatter(df_curr[:-1].x, df_curr[:-1].res, s=20, c='g', label='Train set ' + str(i))

    #for j in range(5):    
    #    plt.scatter(df_curr[-1:].x, df_curr[-1:].res, s=50+5**j, c='green', alpha=0.52-0.12*j, label='_')
    #    plt.scatter(df_curr[-1:].x, df_curr[-1:].predict,  s=50+5**j, c='red', alpha=0.52-0.12*j, label='_')
    #    plt.scatter(df_curr[-1:].x, df_curr[-1:].predict_ridge,  s=50+5**j, c='darkblue', alpha=0.52-0.12*j, label='_')

    #plt.plot(df_curr.x, df_curr.predict, '-', c='r', lw=0.8, ms=5, label='LinReg')
    #plt.plot(df_curr.x, df_curr.predict_ridge, '-', c='darkblue', lw=0.8, ms=5, label='LinReg Ridge')
    #plt.legend(loc='upper left', frameon=True)    
    
#print(len(df.index), len(preds))
print('First 10 predictions:', preds[:10])
print('LR time %.2f sec' % (time.time() - st))
print('Estimated time for full test data: %.2f min' % ((time.time() - st) * int(len(df.index)) / (60 * int(len(preds)))))


# In[62]:

df.loc[:len(preds) - 1, 'preds'] = preds
df['Last'] = df.recursion
df.preds.fillna(df.ints_list.apply(lambda x: x[-1]), inplace=True)
df.Last.fillna(df.preds, inplace=True)
df['Last'] = df['Last'].apply(lambda x: int(np.round(float(x))) if x != np.inf else 1)
print(df[['Id', 'Last']].head(10))
df[['Id', 'Last']].to_csv('submission.csv', sep=',', index=False)

