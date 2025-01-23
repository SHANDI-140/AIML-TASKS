#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("universities.csv")
df


# In[3]:


np.mean(df["SAT"])


# In[4]:


np.median(df["SAT"])


# In[5]:


np.std(df["GradRate"])


# In[6]:


np.var(df["SFRatio"])


# In[7]:


df.describe()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.figure(figsize=(5,5))
plt.title("Graduation Rate")
plt.hist(df["GradRate"])


# # Visualization using bloxpot

# In[11]:


s = [20,15,10,25,30,35,28,40,45,60]
scores = pd.Series(s)
scores


# In[16]:


plt.boxplot(scores, vert=False)


# In[25]:


s = [20,15,10,25,30,35,28,40,45,60,120,150]
scores = pd.Series(s)
scores


# In[26]:


plt.boxplot(scores, vert=False)


# # Identification of outliers in universites data set

# In[31]:


df = pd.read_csv("universities.csv")
df


# In[49]:


#SAT has outliers
plt.figure(figsize=(6,2))
plt.title("Box plot for sat score")
plt.boxplot(df["SAT"],vert=False)


# In[ ]:





# In[ ]:




