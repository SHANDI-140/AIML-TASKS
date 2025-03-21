#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold


# In[2]:


dataframe = pd.read_csv('diabetes.csv')
dataframe


# In[3]:


array = dataframe.values
X = array[:,0:8]
Y = array[:,8]


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y)


# In[8]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[11]:


clf = SVC()
clf.fit(X_train,y_train)


# In[12]:


y_predict = clf.predict(X_test)


# In[13]:


print(classification_report(y_test,y_predict))


# In[15]:


print(classification_report(y_train, clf.predict(X_train)))


# In[16]:


clf = SVC()
param_grid = [{'kernel':['linear','rbf'],'gamma':[0.1,0.5,1],'C':[0.1,1,10]}]
kfold = StratifiedKFold(n_splits=5)
gsv = RandomizedSearchCV(clf,param_grid,cv=kfold,scoring = 'recall')
gsv.fit(X_train,y_train)


# In[17]:


gsv.best_params_, gsv.best_score_


# In[18]:


clf_model = SVC(kernel = 'linear', C=1)
clf_model.fit(X_train,y_train)
y_pred = clf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy=",acc)
confusion_matrix(y_test, y_pred)


# In[19]:


y_pred


# In[ ]:




