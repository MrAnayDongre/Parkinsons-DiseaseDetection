#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Make necessary imports
import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[11]:


#Read the data into data frame
df = pd.read_csv('parkinsons.data')
df.head()


# In[12]:


#Get the deatures and labels
features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


# In[13]:


#Get the count of each lebel as 0 or 1
print(labels[labels==1].shape[0], labels[labels==0].shape[0])


# In[14]:


#Scale the features to between -1 and 1
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels


# In[15]:


#Split the data set 
x_train ,y_train , x_test , y_test = train_test_split(x,y,test_size = .2,random_state = 7)


# In[16]:


#Train the model
#Here XGB classifier is used
model = XGBClassifier()
model.fit(x_train,y_train)


# In[9]:


#Claculate the accuracy
y_predict = model.predict(x_test)
print(accuracy_score(y_test, y_pred)*100)


# In[ ]:




