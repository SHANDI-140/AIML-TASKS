#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[5]:


iris = pd.read_csv("iris.csv")
iris


# In[6]:


iris.info()


# In[7]:


iris.describe()


# In[10]:


iris.isnull()


# In[12]:


iris.std()


# In[13]:


iris.mean()


# In[14]:


iris.median()


# In[15]:


iris.sum()


# In[24]:


iris1=iris.iloc[:,1:]
iris1


# In[25]:


iris[iris1.duplicated(keep = False)]


# 
