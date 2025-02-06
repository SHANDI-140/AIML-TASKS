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


# # Obervations from correlation plots and coeffcients
# 
# Between x and y ,all the x variables are showing moderate tp high correlation
# strengths,highest being between HP and MPG
# 
# Therefor this dataset qualifies for building a multiple linear regression model to predict MPG
# 
# 
# 
# Among x colums (x1,x2,x3 and x4),some veery high correlation strenghts are observed between SP vs HP,VOL vs WT
# 
# 
# 
# The high correlation among x columns is not desirable as it might lead to multicollinearity problem
# 
# 
# 

# # Preparing a preliminary model considering all X columns

# In[14]:


model1=smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[15]:


model1.summary()


# # Observations from model summary 

# The R-squared and adjusted r squared values are good and about 75% of variability in Y is explained by X comlums
# 
# 
# The probability value with respect to F-stastic is close to zero , indicating that all or some of X columns are significant
# 
# The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves which need to be further explored

# # Performance metrics for model1

# In[16]:


df1=pd.DataFrame()
df1["actual_y1"]=cars["MPG"]
df1.head()


# In[18]:


cars.iloc[:,0:4]


# In[19]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[ ]:




