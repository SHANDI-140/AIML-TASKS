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


# In[10]:


data1.rename({'Solar.R':'Solar','Ozone.R':'Ozone'},axis=1,inplace=True)
data1


# # Impute the missing values

# In[11]:


data1.info()


# In[12]:


#Display data1 missing values count in each column using is null.sum()
data1.isnull().sum()


# In[13]:


#Visualize data1 missing values using heat map
cols = data1.columns
colors = ['yellow','blue']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar=True)


# In[14]:


#Find the mean and median values of each numeric
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[15]:


data1["Ozone"]=data1["Ozone"].fillna(median_ozone)
data1.isnull().sum()


# In[16]:


median_Solar = data1["Solar"].median()
mean_Solar = data1["Solar"].mean()
print("Median of Solar: ",median_Solar)
print("Mean of Solar: ",mean_Solar)


# In[17]:


data1["Solar"]=data1["Solar"].fillna(mean_ozone)
data1.isnull().sum()


# In[18]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[20]:


data1["Weather"]=data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[22]:


print(data1["Month"].value_counts())
mode_Month = data1["Month"].mode()[0]
print(mode_Month)


# In[23]:


data1["Month"]=data1["Month"].fillna(mode_Month)
data1.isnull().sum()


# In[25]:


fig,axes = plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["Ozone"],ax=axes[0],color='skyblue',width=0.5,orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1["Ozone"],kde=True,ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# #### Observations
# >The ozone columns has extreme values beyond 81 as seen from box plot
# >The same is confirmed formt the below right-skewed histogram
# 
# 

# In[26]:


fig,axes = plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[1,3]})
sns.boxplot(data=data1["Solar"],ax=axes[0],color='skyblue',width=0.5,orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")
sns.histplot(data1["Solar"],kde=True,ax=axes[1],color='purple',bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# # Observations:
# 
# >The solar columns doesnt have any values that are exceeded
# 
# >The distributions is not perfectly symmetric but slightly left skewed
# 
# 

# In[ ]:




