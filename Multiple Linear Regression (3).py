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


# In[3]:


cars = pd.DataFrame(cars,columns=['HP','VOL','SP','WT','MPG'])
cars


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

# In[4]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[5]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[6]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[7]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='MPG', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='MPG', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# In[8]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15,.85)})

sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')

sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat="density")
ax_hist.set(ylabel='Density')

plt.tight_layout()
plt.show()


# #### Observation from boxplot and histograms
# - There are some extreme values (outliers) observed in towards the night tail of SP and HP distributions
# - In VOL and WT columns a few outliers are observed in both tails of their distributions
# - The extreme values of cars data may have come from the specially designed nature of cars
# - As this is multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while building the regression model

# #### Checking for duplicated rows

# In[9]:


cars[cars.duplicated()]


# #### pair plots and Correlation Coefficients

# In[10]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# - **Pair Plots:** The sns.pairplot() will display scatter plots for each pair of variables in the dataset. This helps to visualize the relationship between different features.

# In[11]:


cars.corr()


# - The correlation_matrix will show the correlation coefficients between the variables in your cars dataset.

# #### Observations from correlation plots and coefficients
# - Between x and y, all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG
# - Therefore this dataset qualiifes for building a multiple linear regression model to predict MPG
# - Among x columns(x1,x2,x3 and x4), some very high correlation strengths are observed between SP vs HP, VOL vs WT
# - The high correlation among x columns is not desirable as it might lead to multicollinearity problem
# 

# #### Prepareing a preliminary model considering all X columns

# In[12]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[13]:


model1.summary()


# #### Observations from model summary
# - The R-squared and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns
# - The probability value with respect to F-statistic is close to zero, indicating that all or someof X columns are significant
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored

# #### Performance metrics for model1

# In[14]:


if1 = pd.DataFrame()
if1["actual_y1"] = cars["MPG"]
if1.head()


# #### Performance metrics for model1

# In[15]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[17]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[19]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"],df1["pred_y1"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# # Checking for multicollinearity among X-colums using VIF method

# In[20]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# # Observations for VIF values:

# The ideal range of VIF values shall be between 0 to 10.However slightly higher valuues can be tolerated
# 
# As seen from the very high VIF values for VOL and WT,it is clear that they are prone to multicollinearity problem.
# 
# Hence it is decided to drop one of the columns(either VOL or WT) to overcome the multicollinearity.
# 
# It is decided to drop WT and retain VOL column in further models

# In[21]:


cars1 = cars.drop("WT",axis=1)
cars1.head()


# In[27]:


model2=smf.ols('MPG~HP+VOL+SP',data=cars1).fit()
model2.summary()


# # Performance metrics fro model2

# In[28]:


df2=pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[29]:


pred_y2 = model2.predict(cars1.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[30]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"],df2["pred_y2"])
print("MSE :",mse)
print("RMSE :",np.sqrt(mse))


# # Observations from model2 summary()
# 
# >The adjusted R-squared value improved slightly tom 0.76
# 
# >All the p-values for model parameters are less than 5% hence they are insignificant
# 
# >Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response variable
# 
# >There is no improvement in MSE value
# 
# 

# In[ ]:




