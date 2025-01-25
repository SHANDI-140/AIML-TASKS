#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("data_clean.csv")
print(data)


# In[3]:


data.info()


# In[4]:


print(type(data))
print(data.shape)
print(data.size)


# In[5]:


data1 = data.drop(['Unnamed: 0','Temp C'],axis =1)
data1


# In[6]:


data1.info()


# In[7]:


data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[8]:


data1[data1.duplicated(keep=False)]


# In[9]:


data1.drop_duplicates(keep='first',inplace=True)
data1


# In[17]:


data1.rename({'Solar.R':'Solar','Ozone.R':'Ozone'},axis=1,inplace=True)
data1


# # Impute the missing values

# In[18]:


data1.info()


# In[19]:


#Display data1 missing values count in each column using is null.sum()
data1.isnull().sum()


# In[27]:


#Visualize data1 missing values using heat map
cols = data1.columns
colors = ['yellow','blue']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[28]:


#Find the mean and median values of each numeric
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[30]:


data1["Ozone"]=data1["Ozone"].fillna(median_ozone)
data1.isnull().sum()


# In[36]:


median_Solar = data1["Solar"].median()
mean_Solar = data1["Solar"].mean()
print("Median of Solar: ",median_Solar)
print("Mean of Solar: ",mean_Solar)


# In[33]:


data1["Solar"]=data1["Solar"].fillna(mean_ozone)
data1.isnull().sum()


# In[ ]:





# In[ ]:




