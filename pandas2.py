#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("universities.csv")
df


# # Sort table by values

# In[4]:


df.sort_values(by="GradRate",ascending=True)


# In[7]:


df[df["GradRate"]>=95]


# In[8]:


df[(df["GradRate"]>80) & (df["SFRatio"]<=12)]


# # Use groupby() to find aggregated values

# In[9]:


sal = pd.read_csv("Salaries.csv")
sal


# In[13]:


sal[["salary","phd","service"]].groupby(sal["rank"]).mean()


# In[ ]:




