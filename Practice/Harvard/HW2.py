
# coding: utf-8

# # Homework 2: More Exploratory Data Analysis
# ## Gene Expression Data and Election Polls 
# 
# Due: Thursday, October 2, 2014 11:59 PM
# 
# <a href=https://raw.githubusercontent.com/cs109/2014/master/homework/HW2.ipynb download=HW2.ipynb> Download this assignment</a>
# 
# #### Submission Instructions
# To submit your homework, create a folder named lastname_firstinitial_hw# and place your IPython notebooks, data files, and any other files in this folder. Your IPython Notebooks should be completely executed with the results visible in the notebook. We should not have to run any code. Compress the folder (please use .zip compression) and submit to the CS109 dropbox in the appropriate folder. If we cannot access your work because these directions are not followed correctly, we will not grade your work.
# 
# 
# ---

# ## Introduction
# 
# John Tukey wrote in [Exploratory Data Analysis, 1977](http://www.amazon.com/Exploratory-Data-Analysis-Wilder-Tukey/dp/0201076160/ref=pd_bbs_sr_2/103-4466654-5303007?ie=UTF8&s=books&qid=1189739816&sr=8-2): "The greatest value of a picture is when it forces us to notice what we never expected to see." In this assignment we will continue using our exploratory data analysis tools, but apply it to new sets of data: [gene expression](http://en.wikipedia.org/wiki/Gene_expression) and polls from the [2012 Presidental Election](http://en.wikipedia.org/wiki/United_States_presidential_election,_2012) and from the [2014 Senate Midterm Elections](http://en.wikipedia.org/wiki/United_States_Senate_elections,_2014).   
# 
# **First**: You will use exploratory data analysis and apply the [singular value decomposition](http://en.wikipedia.org/wiki/Singular_value_decomposition) (SVD) to a gene expression data matrix to determine if the the date that the gene expression samples are processed has large effect on the variability seen in the data.  
# 
# **Second**: You will use the polls from the 2012 Presidential Elections to determine (1) Is there a pollster bias in presidential election polls? and (2) Is the average of polls better than just one poll?
# 
# **Finally**: You will use the [HuffPost Pollster API](http://elections.huffingtonpost.com/pollster/api) to extract the polls for the current 2014 Senate Midterm Elections and provide a preliminary prediction of the result of each state.
# 
# #### Data
# 
# We will use the following data sets: 
# 
# 1. A gene expression data set called `exprs_GSE5859.csv` and sample annotation table called `sampleinfo_GSE5859.csv` which are both available on Github in the 2014_data repository: [expression data set](https://github.com/cs109/2014_data/blob/master/exprs_GSE5859.csv) and [sample annotation table](https://github.com/cs109/2014_data/blob/master/sampleinfo_GSE5859.csv).  
# 
# 2. Polls from the [2012 Presidential Election: Barack Obama vs Mitt Romney](http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama). The polls we will use are from the [Huffington Post Pollster](http://elections.huffingtonpost.com/pollster).  
# 
# 3. Polls from the [2014 Senate Midterm Elections](http://elections.huffingtonpost.com/pollster) from the [HuffPost Pollster API](http://elections.huffingtonpost.com/pollster/api). 
# 
# ---

# ## Load Python modules

# In[293]:

# special IPython command to prepare the notebook for matplotlib
get_ipython().magic('matplotlib inline')

import requests 
from io import StringIO
import numpy as np
import pandas as pd # pandas
import matplotlib.pyplot as plt # module for plotting 
import datetime as dt # module for manipulating dates and times
import numpy.linalg as lin # module for performing linear algebra operations


# ## Problem 1
# 
# In this problem we will be using a [gene expression](http://en.wikipedia.org/wiki/Gene_expression) data set obtained from a [microarray](http://en.wikipedia.org/wiki/DNA_microarray) experiement [Read more about the specific experiment here](http://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE5859).  There are two data sets we will use:  
# 
# 1. The gene expression intensities where the rows represent the features on the microarray (e.g. genes) and the columns represent the different microarray samples.  
# 
# 2. A table that contains the information about each of the samples (columns in the gene expression data set) such as the sex, the age, the treatment status, the date the samples were processed.  Each row represents one sample. 

# #### Problem 1(a) 
# Read in the two files from Github: [exprs_GSE5859.csv](https://github.com/cs109/2014_data/blob/master/exprs_GSE5859.csv) and [sampleinfo_GSE5859.csv](https://github.com/cs109/2014_data/blob/master/sampleinfo_GSE5859.csv) as pandas DataFrames called `exprs` and `sampleinfo`. Use the gene names as the index of the `exprs` DataFrame.

# In[294]:

#url = 'https://github.com/cs109/2014_data/blob/master/exprs_GSE5859.csv'
#df = pd.read_csv(url, sep=',')
import io
url_exprs = "https://raw.githubusercontent.com/cs109/2014_data/master/exprs_GSE5859.csv"
url_sample = "https://raw.githubusercontent.com/cs109/2014_data/master/sampleinfo_GSE5859.csv"
s1 = io.BytesIO(requests.get(url_exprs).content)
s2 = io.BytesIO(requests.get(url_sample).content)
exprs = pd.read_csv(s1)
sampleinfo = pd.read_csv(s2)


# Make sure the order of the columns in the gene expression DataFrame match the order of file names in the sample annotation DataFrame. If the order of the columns the `exprs` DataFrame do not match the order of the file names in the `sampleinfo` DataFrame, reorder the columns in the `exprs` DataFrame. 
# 
# **Note**: The column names of the gene expression DataFrame are the filenames of the orignal files from which these data were obtained.  
# 
# **Hint**: The method `list.index(x)` [[read here](https://docs.python.org/2/tutorial/datastructures.html)] can be used to return the index in the list of the first item whose value is x. It is an error if there is no such item. To check if the order of the columns in `exprs` matches the order of the rows in `sampleinfo`, you can check using the method `.all()` on a Boolean or list of Booleans: 
# 
# Example code: `(exprs.columns == sampleinfo.filename).all()`

# In[295]:

exprs = exprs[list(sampleinfo.filename.unique())]
exprs.columns


# Show the head of the two tables: `exprs` and `sampleinfo`. 

# In[296]:

exprs.head()


# In[297]:

sampleinfo.head()


# #### Problem 1(b)
# 
# Extract the year and month as integers from the `sampleinfo` table. 
# 
# **Hint**: To convert a Series or a column of a pandas DataFrame that contains a date-like object, you can use the `to_datetime` function [[read here](http://pandas.pydata.org/pandas-docs/stable/timeseries.html)].  This will create a `DatetimeIndex` which can be used to extract the month and year for each row in the DataFrame. 

# In[298]:

#your code here
import datetime
sampleinfo['date'] = pd.to_datetime(sampleinfo.date)
sampleinfo['year'] = sampleinfo.date.apply(lambda x: x.year)
sampleinfo['month'] = sampleinfo.date.apply(lambda x: x.month)
sampleinfo['day'] = sampleinfo.date.apply(lambda x: x.day)
sampleinfo.head()


# #### Problem 1(c)
# 
# Convert the dates in the `date` column from the `sampleinfo` table into days since October 31, 2002. Add a column to the `sampleinfo` DataFrame titled `elapsedInDays` containing the days since October 31, 2002.  Show the head of the `sampleinfo` DataFrame which includes the new column.  
# 
# **Hint**: Use the `datetime` module to create a new `datetime` object for the specific date October 31, 2002. Then, subtract the October 31, 2002 date from each date from the `date` column in the `sampleinfo` DataFrame. 

# In[299]:

#your code here
sampleinfo['elapsedInDays'] = sampleinfo.date - datetime.datetime(2002, 10, 31)
sampleinfo.head()


# #### Problem 1(d)
# 
# Use exploratory analysis and the singular value decomposition (SVD) of the gene expression data matrix to determine if the date the samples were processed has large effect on the variability seen in the data or if it is just ethnicity (which is confounded with date). 
# 
# **Hint**: See the end of the [lecture from 9/23/2014 for help with SVD](http://nbviewer.ipython.org/github/cs109/2014/blob/master/lectures/lecture07/data_scraping_transcript.ipynb). 
# 
# First subset the the `sampleinfo` DataFrame to include only the CEU ethnicity.  Call this new subsetted DataFrame `sampleinfoCEU`.  Show the head of `sampleinfoCEU` DataFrame. 

# In[300]:

#your code here
sampleinfoCEU = sampleinfo[sampleinfo.ethnicity == 'CEU']
sampleinfoCEU.head()


# Next, subset the `exprs` DataFrame to only include the samples with the CEU ethnicity. Name this new subsetted DataFrame `exprsCEU`. Show the head of the `exprsCEU` DataFrame. 

# In[301]:

exprsCEU = exprs[[col for col in exprs.columns if col in sampleinfoCEU.filename.unique()]]
exprsCEU.head()


# Check to make sure the order of the columns in the `exprsCEU` DataFrame matches the rows in the `sampleinfoCEU` DataFrame.  

# In[302]:

(exprsCEU.columns == sampleinfoCEU.filename).all()


# Compute the average gene expression intensity in the `exprsCEU` DataFrame across all the samples. For each sample in the `exprsCEU` DataFrame, subtract the average gene expression intensity from each of the samples. Show the head of the mean normalized gene expression data.  

# In[303]:

exprsCEU_norm = exprsCEU.apply(lambda x: x - x.mean())
exprsCEU_norm.head()


# In[304]:

plt.scatter(exprsCEU_norm.ix[3], exprsCEU_norm.ix[10])


# Using this mean normalized gene expression data, compute the projection to the first Principal Component (PC1).  
# 
# **Hint**: Use the `numpy.linalg.svd()` function in the `numpy.linalg` module (or the `scipy.linalg.svd()` function in the `scipy.linalg` module) to apply an [singular value decomposition](http://en.wikipedia.org/wiki/Singular_value_decomposition) to a matrix.  

# In[305]:

U, S, V = np.linalg.svd(exprsCEU_norm)


# In[306]:

svd_pc1 = V[0,:]


# In[307]:

# from sklearn.decomposition import pca
# clf = pca.PCA()
# exprsCEU_norm_pca = clf.fit_transform(exprsCEU_norm.transpose())
# print(clf.explained_variance_ratio_[:5])
# import seaborn as sns
# sns.set(style='ticks', context='notebook')
# sns.distplot(exprsCEU_norm_pca[0], bins=25, kde=False)


# Create a histogram using the values from PC1.  Use a bin size of 25.  

# In[308]:

import seaborn as sns
sns.set(style='ticks', context='notebook')
sns.distplot(svd_pc1, bins=25, kde=False)


# Create a scatter plot with the days since October 31, 2002 on the x-axis and PC1 on the y-axis.
# 
# Around what day do you notice a difference in the way the samples were processed?

# In[309]:

plt.scatter(sampleinfoCEU.elapsedInDays.apply(lambda x: pd.to_timedelta(x).days), svd_pc1)


# Answer: ???

# ## Discussion for Problem 1
# 
# *Write a brief discussion of your conclusions to the questions and tasks above in 100 words or less.*
# 
# I don't know any pros in using SVD. I just don't understand it. Using sklearn.decomposition.PCA seems to be much more clearer.
# 
# ---
# 

# ## Problem 2: Is there a pollster bias in presidential election polls?

# #### Problem 2(a)
# 
# The [HuffPost Pollster](http://elections.huffingtonpost.com/pollster) contains many political polls. You can access these polls from individual races as a CSV but you can also access polls through the [HuffPost Pollster API](http://elections.huffingtonpost.com/pollster/api) to access the data.  
# 
# Read in the polls from the [2012 Presidential Election: Barack Obama vs Mitt Romney](http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama) into a pandas DataFrame called `election`. For this problem, you may read in the polls for this race directly using [the CSV file](http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv) available from the HuffPost Pollster page.

# In[310]:

url = 'http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv'
election = pd.read_csv(url, parse_dates=['Start Date', 'End Date'])


# Show the head of the `election` DataFrame. 

# In[311]:

election.head()


# How many polls were conducted in November? Define this number as M.  
# 
# **Hint**: Subset the `election` DataFrame for only dates in the `Start Date` column that are in November 2012.  

# In[312]:

nov = election[election['Start Date'].apply(lambda x: (x.month == 11) & (x.year == 2012))].drop_duplicates('Pollster')
len(nov.index)


# Answer: 18

# What was the median of the number of observations in the November polls? Define this quantity as N. 

# In[313]:

N = nov['Number of Observations'].median()
print(N)


# Answer: 1200

# #### Problem 2(b)
# 
# Using the median sample size $N$ from Problem 1(a), simulate the results from a single poll:  simulate the number of votes for Obama out of a sample size $N$ where $p$ = 0.53 is the percent of voters who are voting for Obama.  
# 
# **Hint**: Use the binomial distribution with parameters $N$ and $p$ = 0.53. 

# In[314]:

np.random.binomial(n=N, p=0.53)


# Now, perform a Monte Carlo simulation to obtain the estimated percentage of Obama votes with a sample size $N$ where $N$ is the median sample size calculated in Problem 2(a). Let $p$=0.53 be the percent of voters are voting for Obama. 
# 
# **Hint**: You will repeat the simulation above 1,000 times and plot the distribution of the estimated *percent* of Obama votes from a single poll.  The results from the single poll you simulate is random variable and will be different every time you sample. 

# In[315]:

votes = []
for i in range(1000):
    votes.append(np.random.binomial(n=N, p=0.53) / N)
plt.scatter(np.arange(0, 1000), votes)


# Plot the distribution of the estimated percentage of Obama votes from your single poll. What is the distribution of the estimated percentage of Obama votes? 

# In[316]:

sns.distplot(votes, label='Mean = %.2f' % np.mean(votes))
plt.legend(loc='best')


# Answer: normal distribution with mean = 53.02

# What is the standard error (SE) of the estimated percentage from the poll. 
# 
# **Hint**: Remember the SE is the standard deviation (SD) of the distribution of a random variable. 

# In[317]:

print('Standard error = %.5f' % np.std(votes))


# #### Problem 2(c)
# 
# Now suppose we run M polls where M is the number of polls that happened in November (calculated in Problem 2(a)). Run 1,000 simulations and compute the mean of the M polls for each simulation. 

# In[318]:

M = len(nov.index)
per_M = [np.mean(np.random.binomial(n=1, p=0.53, size=1000)) for i in range(M)]


# What is the distribution of the average of polls?
# 
# **Hint**: Show a plot. 

# In[319]:

sns.distplot(per_M, label='Mean = %.2f' % np.mean(per_M))
plt.legend()


# Answer: distribution seems to be normal with mean = 53.00

# What is the standard error (SE) of the average of polls? 

# In[320]:

print('SE = %.5f' % np.std(per_M))


# Answer: standard error = 0.01809

# Is the SE of the average of polls larger, the same, or smaller than that the SD of a single poll (calculated in Problem 2(b))? By how much?
# 
# **Hint**: Compute a ratio of the two quantities.  

# In[321]:

rat = np.std(votes) / np.std(per_M)
print('Ratio = %.4f' % rat)


# Answer: SE of the average polls is 40 times smaller than the SE of a single poll.

# #### Problem 2(d) 
# 
# Repeat Problem 2(c) but now record the *across poll* standard deviation in each simulation. 

# In[322]:

M = len(nov.index)
se = []
for i in range(M):
    votes_per = np.random.binomial(n=N, p=0.53, size=1000) / N
    se.append(np.std(votes_per))
    
print(se)


# What is the distribution of the *across M polls* standard deviation?
# 
# **Hint**: Show a plot. 

# In[323]:

sns.distplot(se, label='Mean = %.5f' % np.mean(se))
plt.legend()


# Answer: distribution seems to be normal with mean = 0.01448.

# #### Problem 2(e) 
# 
# What is the standard deviation of M polls in our real (not simulated) 2012 presidential election data ? 

# In[324]:

print('SE of real polls = %.5f' % (nov.Obama / 100).std())


# Is this larger, the same, or smaller than what we expeced if polls were not biased.

# In[325]:

(nov.Obama / 100).std() / np.std(votes)


# Answer: real SE is a bit smaller than 'synthetic' SE for one poll.

# #### Problem 2(f)
# 
# **For AC209 Students**: Learn about the normal approximation for the binomial distribution and derive the results of Problem 2(b) and 2(c) analytically (using this approximation). Compare the results obtained analytically to those obtained from simulations.

# In[326]:

#your code here


# Answer: 

# ## Discussion for Problem 2
# 
# *Write a brief discussion of your conclusions to the questions and tasks above in 100 words or less.*
# 
# ---
# 

# ## Problem 3: Is the average of polls better than just one poll?

# #### Problem 3(a)
# 
# Most undecided voters vote for one of the two candidates at the election. Therefore, the reported percentages underestimate the final value of both candidates. However, if we assume the undecided will split evenly, then the observed difference should be an unbiased estimate of the final difference. 
# 
# Add a new column to the `election` DataFrame containg the difference between Obama and Romeny called `Diff`. 

# In[327]:

election['Diff'] = election.Obama - election.Romney
election[['Pollster', 'Obama', 'Romney', 'Diff']].head()


# #### Problem 3(b)
# 
# Make a plot of the differences for the week before the election (e.g. 5 days) where the days are on the x-axis and the differences are on the y-axis.  Add a horizontal line showing 3.9%: the difference between Obama and Romney on election day.

# In[328]:

cols = ['Pollster', 'Start Date', 'End Date', 'Obama', 'Romney', 'Diff', 'days_till']
last_day = max(election["Start Date"])
election['days_till'] = (last_day - election['Start Date']).apply(lambda x: pd.to_timedelta(x).days)
week = election[election['Start Date'].apply(lambda x: (last_day - x).days <= 5)].sort_values('days_till')
ax = plt.scatter(week.days_till, week.Diff)
plt.plot(np.linspace(-1, 6, 100), [3.9]*100, 'r--', lw=1.0)
plt.title('Diff between Obama and Romney in the last week')
sns.despine()


# #### Problem 3(c) 
# 
# Make a plot showing the differences by pollster where the pollsters are on the x-axis and the differences on the y-axis. 

# In[329]:

week.Pollster.value_counts()


# Is the *across poll* difference larger than the *between pollster* difference? 

# Answer: 

# #### Problem 3(d)
# 
# Take the average for each pollster and then compute the average of that. Given this difference how confident would you have been of an Obama victory?
# 
# **Hint**: Compute an estimate of the SE of this average based exclusively on the observed data. 

# In[330]:

polls = week.groupby('Pollster').Diff.mean()
print('Obama diff = %.2f +/- %.4f' % (polls.mean(), polls.std()))


# Answer: SE is larger than mean. So, Romney had a chance to win.

# #### Problem 3(e)
# 
# **For AC209 Students**: Show the difference against time and see if you can detect a trend towards the end. Use this trend to see if it improves the final estimate.

# In[334]:

#plt.plot(election[election.days_till < 100].groupby('Start Date')['Diff'].mean())
election[election.days_till < 100].groupby('Start Date')['Diff'].mean().plot()
f = election[election.days_till < 100].groupby('days_till')['Diff'].mean().reset_index()
coef = np.polyfit(f.days_till, f.Diff, deg=5)
p = np.poly1d(coef)
plt.plot(election[election.days_till < 100].groupby('Start Date')['Diff'].mean().index, p(f.days_till), 'r--', lw=1.0)


# Answer: At the beginning of October (one month before election) Romney was slightly better, but then Obama advantage started to rise.

# ## Discussion for Problem 3
# 
# *Write a brief discussion of your conclusions to the questions and tasks above in 100 words or less.*
# 
# ---
# 

# ## Problem 4
# 
# In this last problem, we will use the polls from the [2014 Senate Midterm Elections](http://elections.huffingtonpost.com/pollster) from the [HuffPost Pollster API](http://elections.huffingtonpost.com/pollster/api) to create a preliminary prediction of the result of each state. 
# 
# The HuffPost Pollster API allows you to access the data as a CSV or a JSON response by tacking ".csv" or ".json" at the end of the URLs. For example the 2012 Presidential Election could be accessed as a [.json](http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.json) instead of a [.csv](http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv)

# #### Problem 4(a)
# 
# Read in the polls for **all** of the 2014 Senate Elections using the HuffPost API. For example, we can consider the [2014 Senate race in Kentucky between Mitch McConnell and Alison Grimes](http://elections.huffingtonpost.com/pollster/2014-kentucky-senate-mcconnell-vs-grimes). 
# 
# To search for the 2014 Senate races, use the `topics` parameter in the API [[read more about topics here](http://elections.huffingtonpost.com/pollster/api)].  

# In[ ]:

url_str = "http://elections.huffingtonpost.com/pollster/api/charts/?topic=2014-senate"


# To list all the URLs related to the 2014 Senate races using the pollster API, we can use a list comprehension:

# In[ ]:

election_urls = [election['url'] + '.csv' for election in requests.get(url_str).json()]
election_urls


# Because there so many Senate races, we can create a dictionary of pandas DataFrames that will be keyed by the name of the election (a string). 

# In[ ]:

def build_frame(url):
    """
    Returns a pandas DataFrame object containing
    the data returned from the given url
    """
    source = requests.get(url).text
    
    # Use StringIO because pd.DataFrame.from_csv requires .read() method
    s = StringIO(source)
    
    return pd.DataFrame.from_csv(s, index_col=None).convert_objects(
            convert_dates="coerce", convert_numeric=True)


# In[ ]:

# Makes a dictionary of pandas DataFrames keyed on election string.
dfs = dict((election.split("/")[-1][:-4], build_frame(election)) for election in election_urls)


# Show the head of the DataFrame containing the polls for the 2014 Senate race in Kentucky between McConnell and Grimes.

# In[ ]:

#your code here


# #### Problem 4(b)
# 
# For each 2014 Senate race, create a preliminary prediction of the result for that state.

# In[ ]:

#your code here


# # Submission Instructions
# 
# To submit your homework, create a folder named **lastname_firstinitial_hw#** and place your IPython notebooks, data files, and any other files in this folder. Your IPython Notebooks should be completely executed with the results visible in the notebook. We should not have to run any code.  Compress the folder (please use .zip compression) and submit to the CS109 dropbox in the appropriate folder. *If we cannot access your work because these directions are not followed correctly, we will not grade your work.*
# 

# In[ ]:



