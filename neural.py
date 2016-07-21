
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')


# In[3]:

x1 = np.linspace(0, 100, 50)
y1 = 0.5 * np.linspace(0, 100, 50) + 20 * (np.random.rand(50) - 0.5)
plt.scatter(x1, y1)


# In[4]:

dic = {'x1':x1, 'y':y1}
data = pd.DataFrame(dic)
data.head()


# In[33]:

X_train = data.iloc[:-10][['x1']].as_matrix()
y_train = data.iloc[:-10][['y']].as_matrix()
X_test = data.iloc[-10:][['x1']].as_matrix()
y_test = data.iloc[-10:][['y']].as_matrix()
X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[34]:

import neurolab as nl


# In[40]:

net = nl.net.newp([[0, 100]], 1)
err = net.train(X_train, y_train, epochs=100, show=10, lr=1.0)
#error = net.train(X_train, y_train, epochs=100, show=10, lr=0.5)
#net = nl.net.newp([[0, 1], [0, 1]], 1)
#err = net.train([[0,0], [0,1], [1,0], [1,1]], [[0], [1], [1], [1]], epochs=100, show=10, lr=0.5)


# In[30]:

net.sim([[0,0]])

