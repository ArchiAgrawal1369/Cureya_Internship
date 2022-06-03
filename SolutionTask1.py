#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[2]:


# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('insurance.csv')


# In[3]:


# first 5 rows of the dataframe
insurance_dataset.head()


# In[4]:


# number of rows and columns
insurance_dataset.shape


# In[5]:


# getting some informations about the dataset
insurance_dataset.info()


# In[6]:


# checking for missing values
insurance_dataset.isnull().sum()


# In[7]:


# statistical Measures of the dataset
insurance_dataset.describe()


# In[8]:


sns.barplot(x = 'age', y = 'bmi', data = insurance_dataset)


# In[9]:


sns.lmplot(x = 'age', y = 'bmi', data = insurance_dataset)


# In[10]:


sns.barplot(x = 'region', y = 'charges', data = insurance_dataset)


# In[11]:


sns.lmplot(x = 'children', y = 'age', data = insurance_dataset)


# In[12]:


sns.jointplot(x = 'region', y = 'charges', data = insurance_dataset)


# In[13]:


#Data Pre-Processing

#Encoding the categorical features

# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

3 # encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


# In[14]:


#Splitting the Features and Target
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']


# In[15]:


print(X)


# In[16]:


print(Y)


# In[17]:


#Splitting the data into Training data & Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[18]:


print(X.shape, X_train.shape, X_test.shape)


# In[19]:


#Model Training

# loading the Linear Regression model
regressor = LinearRegression()


# In[20]:


regressor.fit(X_train, Y_train)


# In[21]:


# prediction on training data
training_data_prediction =regressor.predict(X_train)


# In[22]:


# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)


# In[23]:


# prediction on test data
test_data_prediction =regressor.predict(X_test)


# In[24]:


# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)


# In[28]:


input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print('In USD: ',prediction)

cost = prediction * 64



print('The insurance cost is ', cost)

