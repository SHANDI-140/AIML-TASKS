#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


# In[2]:


data1 = pd.read_csv("NewspaperData.csv")
data1


# In[3]:


plt.figure(figsize=(10, 5))
data1.hist()
plt.title('NewspaperData.csv')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()


# In[4]:


plt.scatter(data1["daily"], data1["sunday"])


# ### A HIGH CORRELATION STRENGTH IS OBSEVERED 

# In[5]:


plt.figure(figsize=(6,3))
plt.boxplot(data1["daily"], vert = False)


# In[ ]:




