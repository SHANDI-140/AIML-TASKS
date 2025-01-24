#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[11]:


data.info()


# In[12]:


print(type(data))
print(data.shape)
print(data.size)


# In[15]:


data1 = data.drop(['Unnamed: 0','Temp C'],axis =1)
data1


# In[16]:


data1.info()


# In[17]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[18]:


data1[data1.duplicated(keep=False)]


# In[20]:


data1.drop_duplicates(keep='first',inplace=True)
data1


# In[ ]:




