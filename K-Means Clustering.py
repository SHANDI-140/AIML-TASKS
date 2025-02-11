#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.info()


# - There are no null values
# - This Dataset consists of 7 columns

# In[4]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=Univ, x='SAT', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=Univ, x='SAT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[5]:


# Read all numeric columns in to Univ1
Univ1 = Univ.iloc[:,1:]
Univ1


# In[6]:


cols = Univ1.columns


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[8]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[9]:


clusters_new.labels_


# In[10]:


set(clusters_new.labels_)


# In[11]:


Univ['clusterid_new'] = clusters_new.labels_


# In[12]:


Univ


# In[13]:


Univ.sort_values(by = "clusterid_new")


# In[14]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# ### Observations:
# Cluster 2 appears to be the top rated univerisities cluster as the cutoff score,top10,SFratio paramneter mean values are highest 
# 
# 
# Cluster 1 appears to occupy the middle rated universities
# 
# Cluster 0 comes as the lower level rated universities

# In[17]:


wcss = []
for i in range (1,20):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1,20),wcss)
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


# ### Observations
# 
# we can observe that k = 3 or 4 which indiactes the elbbo2w joints i.e the rate of change of slope decreases

# In[ ]:




