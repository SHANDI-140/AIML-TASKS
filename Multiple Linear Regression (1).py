#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
import numpy as np


# Assumptions in Multilinear Regression
# - **Linearity:** The relationship between the predictors(x) and the response(Y) is linear.
# - **Independence:** Observations are independent of each other.
# - **Homoscedasticity:** The residuals(Y-Y_hat) exhibit constant variance at all levels of the predictor.
# - **Normal Distribution of Errors:** The residuals of the model are normally distributed
# - **No Multicollinearity:** The independent variables should not be too highly correlated with each other

# In[2]:


cars = pd.read_csv("Cars.csv")
cars.head()


# #### Description of columns
# - MPG: Milege of the car (Mile per Gallon)(This is Y-column to be predicted)
# - HP: Horse Power of the car
# - VOL: Volume of the car
# - SP: Top speed of the car
# - WT: Weight of the car

# 
# #### Observations about info(), missing values
# - There are no missing values
# - There are 81 observations(81 different cars data)
# - The data types of columns are also relevant and valid

# In[3]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[4]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[5]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[6]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# OBSERVATIONS FROM BOXPLOT
# 
# There are some extreme values (outliers)observed in towards the right tailof SP and HP distributions
# 

# In[7]:


cars[cars.duplicated()]


# In[8]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# In[9]:


cars.corr()


# In[ ]:





# In[ ]:




