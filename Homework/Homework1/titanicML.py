
# coding: utf-8

# # Titanic ML Program
# 
# ** I wanted to take a basic machine learning approach, as I am new to ML, but I wanted to explore it more**
# **I used, numpy arrays and decision trees to get the data**
# **I removed some of the data that would be difficult to calculate numerically**
# **The program  prints out the array of 0s and 1s for the test data set. Not all of the passengers were included in the dataset because the program drops some that don't have the right amount of info**
# 

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import sklearn

from sklearn import tree
import matplotlib.pyplot as plt


# In[2]:

df = pd.read_csv("train.csv")


# In[3]:

df.head()



# In[4]:

df['Sex']= df['Sex'].map({'male':1, 'female':2})
df['Embarked']= df['Embarked'].map({'S':0, 'C':1,'Q':3})



# In[5]:

predictors = list(df.columns.values) 


# In[6]:

predictors.remove('Name')


# In[7]:

predictors.remove('PassengerId')


# In[8]:

predictors.remove('Ticket')


# In[9]:

predictors.remove('Cabin')
df=df.dropna()
target = df.Survived
predictors.remove('Survived')


# In[10]:

train=df[predictors]



# In[11]:

train


# In[12]:

X=train
Y=target
clf = tree.DecisionTreeClassifier()

clf.fit(X, Y)


# In[13]:

test_df = pd.read_csv("test.csv")
test_df.describe()


# In[14]:

predictors2 = list(test_df.columns.values) 
predictors2.remove('Name')
predictors2.remove('PassengerId')
predictors2.remove('Ticket')
predictors2.remove('Cabin')
test_df = test_df[predictors2]
test_df.describe()


# In[15]:

test_df = test_df.dropna()
test_df.describe()


# In[16]:

test_df=test_df.dropna()
test_df['Sex']= test_df['Sex'].map({'male':1, 'female':2})
test_df['Embarked']= test_df['Embarked'].map({'S':0, 'C':1,'Q':3})
test_df.describe()

test = test_df[predictors2]


# In[17]:

print(clf.predict(test))
print("Size of testing dataset: ")
print(len(test))


# # Looking at test dataset
# 
# A good deal of samples were dropped using `dropna()`.

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



