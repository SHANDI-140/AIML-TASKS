#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np


# In[12]:


df = pd.read_csv("universities.csv")
df


# In[14]:


np.mean(df["SAT"])


# In[15]:


np.median(df["SAT"])


# In[16]:


np.std(df["GradRate"])


# In[17]:


np.var(df["SFRatio"])


# In[18]:


df.describe()


# In[ ]:




