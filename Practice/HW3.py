
# coding: utf-8

# # Homework 3: Prediction and Classification
# 
# Due: Thursday, October 16, 2014 11:59 PM
# 
# <a href=https://raw.githubusercontent.com/cs109/2014/master/homework/HW3.ipynb download=HW3.ipynb> Download this assignment</a>
# 
# #### Submission Instructions
# To submit your homework, create a folder named lastname_firstinitial_hw# and place your IPython notebooks, data files, and any other files in this folder. Your IPython Notebooks should be completely executed with the results visible in the notebook. We should not have to run any code. Compress the folder (please use .zip compression) and submit to the CS109 dropbox in the appropriate folder. If we cannot access your work because these directions are not followed correctly, we will not grade your work.
# 
# ---
# 

# # Introduction
# 
# In this assignment you will be using regression and classification to explore different data sets.  
# 
# **First**: You will use data from before 2002 in the [Sean Lahman's Baseball Database](http://seanlahman.com/baseball-archive/statistics) to create a metric for picking baseball players using linear regression. This is same database we used in Homework 1. This database contains the "complete batting and pitching statistics from 1871 to 2013, plus fielding statistics, standings, team stats, managerial records, post-season data, and more". [Documentation provided here](http://seanlahman.com/files/database/readme2012.txt).
# 
# **Second**: You will use the famous [iris](http://en.wikipedia.org/wiki/Iris_flower_data_set) data set to perform a $k$-neareast neighbor classification using cross validation.  While it was introduced in 1936, it is still [one of the most popular](http://archive.ics.uci.edu/ml/) example data sets in the machine learning community. Wikipedia describes the data set as follows: "The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres." Here is an illustration what the four features measure:
# 
# **Third**: You will investigate the influence of higher dimensional spaces on the classification using another standard data set in machine learning called the The [digits data set](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).  This data set is similar to the MNIST data set discussed in the lecture. The main difference is, that each digit is represented by an 8x8 pixel image patch, which is considerably smaller than the 28x28 pixels from MNIST. In addition, the gray values are restricted to 16 different values (4 bit), instead of 256 (8 bit) for MNIST. 
# 
# **Finally**: In preparation for Homework 4, we want you to read through the following articles related to predicting the 2014 Senate Midterm Elections. 
# 
# * [Nate Silver's Methodology at while at NYT](http://fivethirtyeight.blogs.nytimes.com/methodology/)
# * [How The FiveThirtyEight Senate Forecast Model Works](http://fivethirtyeight.com/features/how-the-fivethirtyeight-senate-forecast-model-works/)
# * [Pollster Ratings v4.0: Methodology](http://fivethirtyeight.com/features/pollster-ratings-v40-methodology/)
# * [Pollster Ratings v4.0: Results](http://fivethirtyeight.com/features/pollster-ratings-v40-results/)
# * [Nate Silver versus Sam Wang](http://www.washingtonpost.com/blogs/plum-line/wp/2014/09/17/nate-silver-versus-sam-wang/)
# * [More Nate Silver versus Sam Wang](http://www.dailykos.com/story/2014/09/09/1328288/-Get-Ready-To-Rumbllllle-Battle-Of-The-Nerds-Nate-Silver-VS-Sam-Wang)
# * [Nate Silver explains critisims of Sam Wang](http://politicalwire.com/archives/2014/10/02/nate_silver_rebuts_sam_wang.html)
# * [Background on the feud between Nate Silver and Sam Wang](http://talkingpointsmemo.com/dc/nate-silver-sam-wang-feud)
# * [Are there swing voters?]( http://www.stat.columbia.edu/~gelman/research/unpublished/swing_voters.pdf)
# 
# 
# 
# ---

# ## Load Python modules

# In[458]:

# special IPython command to prepare the notebook for matplotlib
get_ipython().magic('matplotlib inline')

import requests 
import io
import zipfile
import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting 

# If this module is not already installed, you may need to install it. 
# You can do this by typing 'pip install seaborn' in the command line
import seaborn as sns 

import sklearn
import sklearn.datasets
import sklearn.cross_validation
import sklearn.decomposition
import sklearn.grid_search
import sklearn.neighbors
import sklearn.metrics


# # Problem 1: Sabermetrics
# 
# Using data preceding the 2002 season pick 10 offensive players keeping the payroll under $20 million (assign each player the median salary). Predict how many games this team would win in a 162 game season.  
# 
# In this problem we will be returning to the [Sean Lahman's Baseball Database](http://seanlahman.com/baseball-archive/statistics) that we used in Homework 1.  From this database, we will be extract five data sets containing information such as yearly stats and standing, batting statistics, fielding statistics, player names, player salaries and biographical information. You will explore the data in this database from before 2002 and create a metric for picking players. 

# #### Problem 1(a) 
# 
# Load in [these CSV files](http://seanlahman.com/files/database/lahman-csv_2014-02-14.zip) from the [Sean Lahman's Baseball Database](http://seanlahman.com/baseball-archive/statistics). For this assignment, we will use the 'Teams.csv', 'Batting.csv', 'Salaries.csv', 'Fielding.csv', 'Master.csv' tables. Read these tables into separate pandas DataFrames with the following names. 
# 
# CSV file name | Name of pandas DataFrame
# :---: | :---: 
# Teams.csv | teams
# Batting.csv | players
# Salaries.csv | salaries
# Fielding.csv | fielding
# Master.csv | master

# In[459]:

teams = pd.read_csv('./input/Teams.csv')
players = pd.read_csv('./input/Batting.csv')
salaries = pd.read_csv('./input/Salaries.csv')
fielding = pd.read_csv('./input/Fielding.csv')
master = pd.read_csv('./input/Master.csv')


# #### Problem 1(b)
# 
# Calculate the median salary for each player and create a pandas DataFrame called `medianSalaries` with four columns: (1) the player ID, (2) the first name of the player, (3) the last name of the player and (4) the median salary of the player. Show the head of the `medianSalaries` DataFrame.   

# In[460]:

master['median_sal'] = master.playerID.map(salaries.groupby('playerID').salary.median())
medianSalaries = master[['playerID', 'nameFirst', 'nameLast', 'median_sal']]
medianSalaries.head()


# In[461]:

pl_stat_cols = ['R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP']
pl_cols = ['playerID', 'yearID', 'teamID'] + pl_stat_cols
for col in pl_stat_cols:
    players[col] = players[col] / players.G
players[(players.yearID == 2001) & (players.teamID == 'OAK')].sort_values('G', ascending=False)[pl_cols].head()


# In[462]:

f_stat_cols = ['PO', 'A', 'E', 'DP', 'PB', 'WP', 'SB', 'CS', 'ZR']
for col in f_stat_cols:
    fielding[col] = fielding[col] / fielding.G
f_cols = ['playerID', 'yearID', 'teamID'] + stat_cols
fielding[(fielding.yearID == 2001) & (fielding.teamID == 'OAK')].sort_values('G', ascending=False)[f_cols].head()


# In[463]:

total_stats = pd.merge(players[pl_cols], fielding[f_cols], on=['playerID', 'yearID', 'teamID'])
total_stats.fillna(0, inplace=True)
total_stats = total_stats[total_stats.yearID < 2002].sort_values(['yearID', 'teamID'])
total_stats.head()


# In[464]:

team_stats = total_stats[total_stats.yearID >= 1962].groupby(['yearID', 'teamID']).mean()
team_stats.head()


# In[465]:

ts = team_stats.join(teams.groupby(['yearID', 'teamID']).W.mean())


# In[466]:

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(ts.iloc[:,:-1], ts.iloc[:, -1])
ts.iloc[:,:-1].shape, ts.iloc[:, -1].shape


# In[467]:

medianSalaries = medianSalaries[medianSalaries.median_sal.notnull()].sort_values('median_sal', ascending=False)
salaries10 = medianSalaries[medianSalaries.median_sal < 1000000].head(20)
ts_cols = ['R', 'H', '2B', '3B', 'HR', 'RBI', 'SB_x', 'CS_x', 'BB', 'SO', 'IBB',
       'HBP', 'SH', 'SF', 'GIDP', 'PO', 'A', 'E', 'DP', 'PB', 'WP', 'SB_y',
       'CS_y', 'ZR']


# In[468]:

salaries10_corr = salaries10.set_index('playerID')    .join(total_stats.drop(['yearID', 'teamID'], axis=1).groupby('playerID').mean())    .dropna(subset=[ts_cols]).head(10)
print('Picked players (payroll = %d):' % salaries10_corr.median_sal.sum())
salaries10_corr


# In[469]:

#clf.predict(salaries10_corr[ts_cols].mean())
clf.predict(salaries10_corr[ts_cols].mean().reshape(1, -1))


# #### Problem 1(c)
# 
# Now, consider only team/season combinations in which the teams played 162 Games. Exclude all data from before 1947. Compute the per plate appearance rates for singles, doubles, triples, HR, and BB. Create a new pandas DataFrame called `stats` that has the teamID, yearID, wins and these rates.
# 
# **Hint**: Singles are hits that are not doubles, triples, nor HR. Plate appearances are base on balls plus at bats.

# In[470]:

teams['1B'] = teams.H - teams['2B'] - teams['3B'] - teams.HR
teams['PA'] = teams.BB + teams.AB
stats = teams[teams.yearID >= 1962][['teamID', 'yearID', 'PA', '1B', '2B', '3B', 'HR', 'BB', 'W']].copy(deep=True)
cols = ['1B', '2B', '3B', 'HR', 'BB']
for col in cols:
    stats[col] = stats[col] / stats.PA
    
stats.head()


# #### Problem 1(d)
# 
# Is there a noticeable time trend in the rates computed computed in Problem 1(c)? 

# In[471]:

import seaborn as sns
sns.set(style='whitegrid', context='talk')
plt.subplots(2, 3, figsize=(20, 10))
cnt = 0
for col in cols:
    cnt += 1
    plt.subplot(2, 3, cnt)
    plt.scatter(stats.yearID, stats[col], label=col, alpha=0.5)
    coef = np.polyfit(stats.yearID, stats[col], deg=5)
    p = np.poly1d(coef)
    plt.plot(stats.yearID, p(stats.yearID), 'r--')
    plt.legend()    


# #### Problem 1(e) 
# 
# Using the `stats` DataFrame from Problem 1(c), adjust the singles per PA rates so that the average across teams for each year is 0. Do the same for the doubles, triples, HR, and BB rates. 

# In[472]:

from sklearn.preprocessing import scale
gs = stats.groupby('yearID')
for col in cols:
    for y in stats.yearID.unique():
        stats.loc[stats.yearID == y, col] =  scale(stats[stats.yearID == y][col])
stats.head()


# #### Problem 1(f)
# 
# Build a simple linear regression model to predict the number of wins from the average adjusted singles, double, triples, HR, and BB rates. To decide which of these terms to include fit the model to data from 2002 and compute the average squared residuals from predictions to years past 2002. Use the fitted model to define a new sabermetric summary: offensive predicted wins (OPW). Hint: the new summary should be a linear combination of one to five of the five rates.
# 

# In[473]:

from sklearn.linear_model import LinearRegression

X_train = stats[stats.yearID < 2002][cols]
y_train = stats[stats.yearID < 2002].W
X_test = stats[stats.yearID >= 2002][cols]
y_test = stats[stats.yearID >= 2002].W
clf = LinearRegression()
clf.fit(X_train, y_train)
stats['OPW'] = np.round(clf.predict(stats[cols]))
stats[stats.yearID >= 2002].head()
print('Average squared residuals = %.2f' % np.mean((y_test - clf.predict(X_test)) ** 2))
print('Coefs:', np.round(clf.coef_, 2))


# ** Your answer here: **

# #### Problem 1(g)
# 
# Now we will create a similar database for individual players. Consider only player/year combinations in which the player had at least 500 plate appearances. Consider only the years we considered for the calculations above (after 1947 and seasons with 162 games). For each player/year compute singles, doubles, triples, HR, BB per plate appearance rates. Create a new pandas DataFrame called `playerstats` that has the playerID, yearID and the rates of these stats.  Remove the average for each year as for these rates as done in Problem 1(e). 

# In[474]:

players = pd.read_csv('./input/Batting.csv')
players['PA'] = players.AB + players.BB
players['1B'] = players.H - players['2B'] - players['3B'] - players.HR
cols = ['1B', '2B', '3B', 'HR', 'BB']
playerstats = players[(players.yearID >= 1962) & (players.PA >= 500)][['playerID', 'yearID', 'teamID', 'PA'] + cols]        .sort_values(['yearID', 'teamID'])


# Show the head of the `playerstats` DataFrame. 

# In[475]:

from sklearn.preprocessing import scale
gs = playerstats.groupby('yearID')
for col in cols:
    for y in playerstats.yearID.unique():
        playerstats.loc[playerstats.yearID == y, col] =  scale(playerstats[playerstats.yearID == y][col])
playerstats.head()


# #### Problem 1(h)
# 
# Using the `playerstats` DataFrame created in Problem 1(g), create a new DataFrame called `playerLS` containing the player's lifetime stats. This DataFrame should contain the playerID, the year the player's career started, the year the player's career ended and the player's lifetime average for each of the quantities (singles, doubles, triples, HR, BB). For simplicity we will simply compute the avaerage of the rates by year (a more correct way is to go back to the totals). 

# In[476]:

playerLS = pd.DataFrame(playerstats.playerID.unique(), columns=['playerID'])
playerLS['year_start'] = playerLS.playerID.map(players.groupby('playerID').yearID.min())
playerLS['year_end'] = playerLS.playerID.map(players.groupby('playerID').yearID.max())
playerLS = playerLS.set_index('playerID').join(playerstats.groupby('playerID')[cols].mean())


# Show the head of the `playerLS` DataFrame. 

# In[477]:

playerLS.head()


# #### Problem 1(i)
# 
# Compute the OPW for each player based on the average rates in the `playerLS` DataFrame. You can interpret this summary statistic as the predicted wins for a team with 9 batters exactly like the player in question. Add this column to the playerLS DataFrame. Call this column OPW.

# In[478]:

playerLS['OPW'] = clf.predict(playerLS[cols])
playerLS = playerLS.reset_index()
playerLS.head()


# #### Problem 1(j)
# 
# Add four columns to the `playerLS` DataFrame that contains the player's position (C, 1B, 2B, 3B, SS, LF, CF, RF, or OF), first name, last name and median salary. 

# In[479]:

playerLS['pos'] = playerLS.playerID.map(fielding.drop_duplicates('playerID').set_index('playerID').POS)
if 'median_sal' not in playerLS.columns:
    playerLS = pd.merge(playerLS, medianSalaries, on='playerID')


# Show the head of the `playerLS` DataFrame. 

# In[480]:

playerLS.head()


# #### Problem 1(k)
# 
# Subset the `playerLS` DataFrame for players active in 2002 and 2003 and played at least three years. Plot and describe the relationship bewteen the median salary (in millions) and the predicted number of wins. 

# In[542]:

sns.set_context('notebook')
playerLS3 = playerLS[(playerLS.year_start <= 2002) & (playerLS.year_end >= 2003) 
                     & (playerLS.year_end - playerLS.year_start >= 2)].copy()
plt.scatter(playerLS3.median_sal / 1e6, playerLS3.OPW)
plt.xlabel('Salary, mln')
plt.ylabel('Predicted wins')


# #### Problem 1(l)
# Pick one players from one of each of these 10 position C, 1B, 2B, 3B, SS, LF, CF, RF, DH, or OF keeping the total median salary of all 10 players below 20 million. Report their averaged predicted wins and total salary.

# In[544]:

p = np.poly1d(np.polyfit(playerLS3.median_sal, playerLS3.OPW, deg=1))
playerLS3['res'] = playerLS3.OPW - p(playerLS3.median_sal)
cols = ['nameFirst', 'nameLast', 'pos', 'OPW', 'median_sal', 'res']


# In[556]:

def get_max_salary(sl, sp):        
    return sl - playerLS3[playerLS3.pos.isin(sp) == False].groupby('pos').median_sal.min().sum()

salary_left = 20 * 10**6
pos = sel.pos.unique()
sel_pos = []
sel_pl = []
for p in pos:    
    msal = get_max_salary(salary_left, sel_pos)
    pl = playerLS3[(playerLS3.median_sal <= msal) & (playerLS3.pos == p)].res.idxmax()    
    sel_pos.append(p)
    sel_pl.append(pl)
    salary_left = salary_left - playerLS3.ix[pl].median_sal
sel = playerLS3.ix[sel_pl]
print(sel.median_sal.sum())
print(sel[cols])
print(sel.OPW.mean())


# #### Problem 1(m)
# What do these players outperform in? Singles, doubles, triples HR or BB?

# In[562]:

playerLS3.ix[sel_pl][['playerID', '1B', '2B', '3B', 'HR', 'BB']].mean()


# ** Your answer here: **
# Selected players outperformed in BB. Average BB of selected team is much bigger than average.

# ## Discussion for Problem 1
# 
# *Write a brief discussion of your conclusions to the questions and tasks above in 100 words or less.*
# 
# ---

# # Problem 2:  $k$-Nearest Neighbors and Cross Validation 
# 
# What is the optimal $k$ for predicting species using $k$-nearest neighbor classification 
# on the four features provided by the iris dataset.
# 
# In this problem you will get to know the famous iris data set, and use cross validation to select the optimal $k$ for a $k$-nearest neighbor classification. This problem set makes heavy use of the [sklearn](http://scikit-learn.org/stable/) library. In addition to Pandas, it is one of the most useful libraries for data scientists! After completing this homework assignment you will know all the basics to get started with your own machine learning projects in sklearn. 
# 
# Future lectures will give further background information on different classifiers and their specific strengths and weaknesses, but when you have the basics for sklearn down, changing the classifier will boil down to exchanging one to two lines of code.
# 
# The data set is so popular, that sklearn provides an extra function to load it:

# In[564]:

#load the iris data set
iris = sklearn.datasets.load_iris()

X = iris.data  
Y = iris.target

print(X.shape, Y.shape)


# In[618]:

pd.tools.plotting.scatter_matrix(pd.DataFrame(X), figsize=(10, 8))
plt.show()


# #### Problem 2(a) 
# Split the data into a train and a test set. Use a random selection of 33% of the samples as test data. Sklearn provides the [`train_test_split`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html) function for this purpose. Print the dimensions of all the train and test data sets you have created. 

# In[574]:

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.33)
X_train.shape, X_test.shape


# #### Problem 2(b)
# 
# Examine the data further by looking at the projections to the first two principal components of the data. Use the [`TruncatedSVD`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) function for this purpose, and create a scatter plot. Use the colors on the scatter plot to represent the different classes in the target data. 

# In[621]:

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2)
#X_scaled = X_train - np.mean(X_train, axis=0)
X_scaled = X - np.mean(X, axis=0)
svd.fit(X_scaled)
print(svd.explained_variance_ratio_)
X_svd = svd.transform(X_scaled)
df_svd = pd.DataFrame(X_svd, columns=['pc1', 'pc2'])
#df_svd['res'] = y_train
df_svd['res'] = Y
plt.scatter(df_svd[df_svd.res == 0].pc1, df_svd[df_svd.res == 0].pc2)
plt.scatter(df_svd[df_svd.res == 1].pc1, df_svd[df_svd.res == 1].pc2, c='r')
plt.scatter(df_svd[df_svd.res == 2].pc1, df_svd[df_svd.res == 2].pc2, c='g')


# #### Problem 2(c) 
# 
# In the lecture we discussed how to use cross validation to estimate the optimal value for $k$ (the number of nearest neighbors to base the classification on). Use ***ten fold cross validation*** to estimate the optimal value for $k$ for the iris data set. 
# 
# **Note**: For your convenience sklearn does not only include the [KNN classifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), but also a [grid search function](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV). The function is called grid search, because if you have to optimize more than one parameter, it is common practice to define a range of possible values for each parameter. An exhaustive search then runs over the complete grid defined by all the possible parameter combinations. This can get very computation heavy, but luckily our KNN classifier only requires tuning of a single parameter for this problem set. 

# In[677]:

from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, {'n_neighbors':np.arange(1, 21)})
clf.fit(X_train, y_train)


# #### Problem 2(d)
# 
# Visualize the result by plotting the score results versus values for $k$. 

# In[678]:

res = [score.cv_validation_scores for i, score in enumerate(clf.grid_scores_)]
plt.scatter(np.arange(1, 21), np.mean(res, axis=1), c='black')
plt.boxplot(res)
plt.ylim(0.85, 1.02)
plt.show()


# Verify that the grid search has indeed chosen the right parameter value for $k$.

# In[688]:

best_n = clf.best_estimator_.n_neighbors
clf.best_params_


# #### Problem 2(e)
# 
# Test the performance of our tuned KNN classifier on the test set.

# In[715]:

knn = KNeighborsClassifier(n_neighbors=best_n)
from sklearn.cross_validation import KFold
kf = KFold(n=X_scaled.shape[0], n_folds=5, shuffle=True)
sc = []
for train_index, test_index in kf:
    X_train = X_svd[train_index]
    y_train = Y[train_index]
    X_test = X_svd[test_index]
    y_test = Y[test_index]
    knn.fit(X_train, y_train)
    sc.append(knn.score(X_test, y_test))    
    
print(np.round(np.mean(sc), 2), np.round(np.std(sc), 2))


# ## Discussion for Problem 2
# 
# *Write a brief discussion of your conclusions to the questions and tasks above in 100 words or less.*
# 
# ---

# # Problem 3: The Curse and Blessing of Higher Dimensions
# 
# In this problem we will investigate the influence of higher dimensional spaces on the classification. The data set is again one of the standard data sets from sklearn. The [digits data set](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) is similar to the MNIST data set discussed in the lecture. The main difference is, that each digit is represented by an 8x8 pixel image patch, which is considerably smaller than the 28x28 pixels from MNIST. In addition, the gray values are restricted to 16 different values (4 bit), instead of 256 (8 bit) for MNIST. 
# 
# First we again load our data set.

# In[717]:

digits = sklearn.datasets.load_digits()

X = digits.data  
Y = digits.target

print(X.shape, Y.shape)


# #### Problem 3(a) 
# 
# Start with the same steps as in Problem 2. Split the data into train and test set. Use 33% of the samples as test data. Print the dimensions of all the train and test data sets you created. 

# In[721]:

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.33)
print(X_train.shape, X_test.shape)


# In[736]:

plt.subplots(5, 5, figsize=(5,5))
for cnt in np.arange(1, 26):
    plt.subplot(5, 5, cnt)
    plt.imshow(X[cnt-1].reshape(8, 8))


# #### Problem 3(b) 
# 
# Similar to Problem 2(b), create a scatter plot of the projections to the first two PCs.  Use the colors on the scatter plot to represent the different classes in the target data. How well can we separate the classes?
# 
# **Hint**: Use a `Colormap` in matplotlib to represent the diferent classes in the target data. 

# In[763]:

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X - np.mean(X, axis=0))
print(svd.explained_variance_ratio_)
plt.scatter(X_svd[:,0], X_svd[:,1], c=Y, cmap='coolwarm')
plt.colorbar()
plt.show()


# Create individual scatter plots using only two classes at a time to explore which classes are most difficult to distinguish in terms of class separability.  You do not need to create scatter plots for all pairwise comparisons, but at least show one. 

# In[766]:

df_svd = pd.DataFrame(X_svd, columns=['pc1', 'pc2'])
df_svd['res'] = Y
plt.subplots(3, 3, figsize=(8, 8))
digit = 2
plt.suptitle('Scatter plots for digit = %d' % digit)
for i in range(0, 9):
    plt.subplot(3, 3, i + 1)
    sec = i if i < digit else i + 1
    plt.scatter(df_svd[df_svd.res == 1].pc1, df_svd[df_svd.res == 1].pc2, c='b')
    plt.scatter(df_svd[df_svd.res == sec].pc1, df_svd[df_svd.res == sec].pc2, c='r')
    plt.title(sec)
    
plt.tight_layout()


# Give a brief interpretation of the scatter plot. Which classes look like hard to distinguish? Do both feature dimensions contribute to the class separability? 

# ** Your answer here: **
# 
# It is very difficult to distinguish 2 from 5 and from 8.

# #### Problem 3(c) 
# 
# Write a **ten-fold cross validation** to estimate the optimal value for $k$ for the digits data set. *However*, this time we are interested in the influence of the number of dimensions we project the data down as well. 
# 
# Extend the cross validation as done for the iris data set, to optimize $k$ for different dimensional projections of the data. Create a boxplot showing test scores for the optimal $k$ for each $d$-dimensional subspace with $d$ ranging from one to ten. The plot should have the scores on the y-axis and the different dimensions $d$ on the x-axis. You can use your favorite plot function for the boxplots. [Seaborn](http://web.stanford.edu/~mwaskom/software/seaborn/index.html) is worth having a look at though. It is a great library for statistical visualization and of course also comes with a [`boxplot`](http://web.stanford.edu/~mwaskom/software/seaborn/generated/seaborn.boxplot.html) function that has simple means for changing the labels on the x-axis.

# In[774]:

get_ipython().magic('pinfo GridSearchCV')


# In[865]:

dims = np.arange(1, 11)
ns = np.arange(2, 15)
res = {}
arr = []
best = []
for dim in dims:
    svd = TruncatedSVD(n_components=dim)
    X_svd = svd.fit_transform(X - np.mean(X, axis=0))
    kf = KFold(n=X_svd.shape[0], n_folds=10, shuffle=True)    
    knn = KNeighborsClassifier()
    clf = GridSearchCV(knn, {'n_neighbors':ns}, cv=kf)
    clf.fit(X_svd, Y)
    res = [score.cv_validation_scores for i, score in enumerate(clf.grid_scores_)]
    rv = np.array(res).ravel()
    arr.append(rv)
    best.append(clf.best_params_['n_neighbors'])    


# In[888]:

plt.boxplot(arr, labels=['d=%d,\nk=%d' % (d, best[d-1]) for d in dims])
plt.show()


# 
# Write a short interpretation of the generated plot, answering the following questions:
# 
# * What trend do you see in the plot for increasing dimensions?
# 
# * Why do you think this is happening?

# ** Your answer here: **
# 
# Score of knn-classifier increases with increasing number or principal components. It happens because little number of components cannot explain enough variance of initial data. We cannot separate one class from another using little number of components (see scatter plots above). Nine components seem to be enough.
# 
# ** Good answer ** 
# 
# The accuracy gets better with increasing dimensions. We have enough data points that the curse of dimensionality does not harm our predictions here and the additional dimensions add to the class separability. There are two factors influencing this. One is that we retain more information of the original signal if we reduce the dimensionality of the data less. The other factor is that the higher dimensional space provides more room between data points and thus more flexibility for the classifier.

# In[889]:

svd = TruncatedSVD(n_components=10)
X_svd = svd.fit_transform(X - np.mean(X, axis=0))
kf = KFold(n=X_svd.shape[0], n_folds=10, shuffle=True)    
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, {'n_neighbors':ns}, cv=kf)
clf.fit(X_svd, Y)
res = [score.cv_validation_scores for i, score in enumerate(clf.grid_scores_)]
plt.boxplot(res)
plt.show()
print(clf.best_params_)


# In[891]:

svd = TruncatedSVD(n_components=10)
X_svd = svd.fit_transform(X - np.mean(X, axis=0))
knn = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_svd, Y, test_size=0.33)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)


# In[892]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80, max_depth=10, max_leaf_nodes=150)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


# #### Problem 3(d) 
# 
# **For AC209 Students**: Change the boxplot we generated above to also show the optimal value for $k$ chosen by the cross validation grid search. 

# In[893]:

plt.boxplot(arr, labels=['d=%d,\nk=%d' % (d, best[d-1]) for d in dims])
plt.show()


# Write a short interpretation answering the following questions:
# 
# * Which trend do you observe for the optimal value of $k$?
# 
# * Why do you think this is happening?

# ** Your answer here: **
# 
# Optimal value of $k$ become smaller. This happens because out points become more seperable in higher dimensional space.

# ## Discussion for Problem 3
# 
# *Write a brief discussion of your conclusions to the questions and tasks above in 100 words or less.*
# 
# ---

# # Submission Instructions
# 
# To submit your homework, create a folder named **lastname_firstinitial_hw#** and place your IPython notebooks, data files, and any other files in this folder. Your IPython Notebooks should be completely executed with the results visible in the notebook. We should not have to run any code.  Compress the folder (please use .zip compression) and submit to the CS109 dropbox in the appropriate folder. *If we cannot access your work because these directions are not followed correctly, we will not grade your work.*
# 

# In[31]:



