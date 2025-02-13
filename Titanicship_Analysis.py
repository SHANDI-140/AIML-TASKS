#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[4]:


titanic.info()


# ### Observations
# there are no null values
# 
# these are object and catrgorical in nature
# 
# As the columns are categorical we can adopt one-hot-encoding

# In[9]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[16]:


counts = titanic['Age'].value_counts()
plt.bar(counts.index, counts.values)


# ### Observations
# 
# There are more adults than the children

# In[17]:


counts = titanic['Survived'].value_counts()
plt.bar(counts.index, counts.values)


# ### Observations
# 
# The people who are survived is less than who are dead

# In[18]:


df=pd.get_dummies(titanic,dtype=int)
df.head()


# In[19]:


df.info()


# #### Apriori Algorithm 

# In[21]:


frequent_itemsets=apriori(df,min_support = 0.05,use_colnames = True,max_len=None)
frequent_itemsets


# In[22]:


frequent_itemsets.info()


# In[24]:


#Generate associatin rules with metrics
rules = association_rules(frequent_itemsets,metric="lift", min_threshold=1.0)
rules


# In[25]:


rules.sort_values(by='lift',ascending=True)


# In[ ]:




