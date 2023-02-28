#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Name : Amodi Baakle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("CarPrice.csv")
data.head()


# In[15]:


print(data.isnull().sum())


# In[16]:


print(data.info)


# In[17]:


print(data.describe())


# In[18]:


data.CarName.unique()


# In[19]:


data.fueltype.unique()


# In[20]:


data.aspiration.unique()


# In[21]:


data.carbody.unique()


# In[22]:


sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))
sns.distplot(data.price)
plt.show()


# In[23]:


#displaying the heat map
plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="crest", annot=True)
plt.show()


# In[24]:


predict = "price"
data = data[["symboling", "wheelbase", "carlength", 
             "carwidth", "carheight", "curbweight", 
             "enginesize", "boreratio", "stroke", 
             "compressionratio", "horsepower", "peakrpm", 
             "citympg", "highwaympg", "price"]]
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


# In[25]:


xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)


# In[26]:


model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)


# In[27]:


model.score(xtest, predictions)


# In[28]:


print("Accuracy = ",100*model.score(xtest, predictions))


# In[ ]:




