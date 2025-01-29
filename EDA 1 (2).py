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


# In[19]:


data1["Weather"]=data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[20]:


print(data1["Month"].value_counts())
mode_Month = data1["Month"].mode()[0]
print(mode_Month)


# In[21]:


data1["Month"]=data1["Month"].fillna(mode_Month)
data1.isnull().sum()


# In[22]:


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

# In[23]:


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

# In[24]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"],vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# # USING mu +/-3*SIGMA LIMITS (STANDARD DEVIATION METHOD)

# In[25]:


data1["Ozone"].describe()


# In[26]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# # Observations
# It is observerd tjeat only two outliers are identified using std method
# In boxplot method more no of outliers are identified

# # Quantile-QUantile plot detection of outliers

# In[27]:


import scipy.stats as stats

plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"],dist="norm",plot=plt)
plt.title("Q-Q Pllot for Outlier Detection",fontsize=14)
plt.xlabel("Theoretical Quantiles",fontsize=12)


# # Observations from Q-Q plot
# >The data does not follow normal distribution as the data points are deviating signifacantly away from the red line
# 
# 
# 
# 
# >The data shows a right-skewed distribution and possible outliers

# In[28]:


sns.violinplot(data=data1["Ozone"],color="lightgreen")
plt.title("Violin Plot")
plt.show()


# In[29]:


sns.swarmplot(data=data1, x="Weather",y="Ozone",color="orange",palette="Set2",size=6)


# In[30]:


sns.stripplot(data=data1, x="Weather",y="Ozone",color="orange",palette="Set1",size=6,jitter=True)


# In[31]:


sns.kdeplot(data=data1["Ozone"],fill=True,color="red")
sns.rugplot(data=data1["Ozone"],color="black")


# In[32]:


sns.boxplot(data=data1,x="Weather",y="Ozone")


# # Correlation coefficient and pair plots

# In[34]:


plt.scatter(data1["Wind"],data1["Temp"])


# In[35]:


#Compute pearson correlation coefficient
data1["Wind"].corr(data1["Temp"])


# In[37]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# In[ ]:




