#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation using Clustering

# # Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
import math
import seaborn as sns


# # Data Collection & Analysis

# In[3]:


train = pd.read_csv('C:/Users/harshit/Desktop/Cureya/week 3/Train.csv')
test = pd.read_csv('C:/Users/harshit/Desktop/Cureya/week 3/Test.csv')


# In[4]:


train.select_dtypes('object').columns


# In[5]:


train.info()


# In[6]:


cat_var = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score','Var_1']


# In[7]:


num_var = ['Age','Work_Experience','Family_Size']


# In[8]:


train['Segmentation'].value_counts()


# In[9]:


plt.figure(figsize = (12,8))
sns.countplot(train['Segmentation'])
plt.show()


# # Age vs Work_Experience

# In[10]:


sns.scatterplot(x = 'Work_Experience',y='Age',data = train)


# # Age vs Family_size

# In[11]:


sns.scatterplot(x = 'Family_Size',y='Age',data = train)


# # Age , Work_experience, Family_size

# In[12]:


train[num_var].hist(figsize =(20,10))


# # Correlation Matrix

# In[13]:


plt.figure(figsize = (12,8))
sns.heatmap(train.corr(),annot = True)
plt.show()


# #  categorical data

# In[14]:


# Train data
dummies = pd.get_dummies(data = train[cat_var])
train = pd.concat([train,dummies],axis = 1)
train.drop(['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score','Var_1'],axis = 1,inplace = True)

# Test data
dummies_test = pd.get_dummies(data = test[['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score','Var_1']])
test = pd.concat([test,dummies_test],axis = 1)
test.drop(['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score','Var_1'],axis = 1,inplace = True)


# # Scaling the numeric variables

# In[15]:


from sklearn.preprocessing import MinMaxScaler


# In[16]:


scaler = MinMaxScaler()


# In[17]:


def scaleColumns(df, cols_to_scale):
    for col in cols_to_scale:
        df[col] = scaler.fit_transform(pd.DataFrame(df[col]))
    return df


# In[18]:


train = scaleColumns(train,num_var)

test = scaleColumns(test,num_var)


# In[19]:


train.head()


# # Label encoding target variable

# In[1]:


from sklearn.preprocessing import LabelEncoder


# In[20]:


le = LabelEncoder()
train['Segmentation'] = le.fit_transform(train['Segmentation'])


# In[21]:


train.head()


# # Train Test Split

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


target =['Segmentation']
IDcol = ['ID']

features = [x for x in train.columns if x not in target+IDcol]


# In[ ]:





# In[24]:


X = train[features].values
y= train[target].values


# In[25]:


X.shape,y.shape


# In[26]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42,stratify =y)


# In[27]:


X_train.shape,X_test.shape


# # KNN

# In[28]:


from sklearn.neighbors import KNeighborsClassifier


# In[29]:


# WE will check the accuracy scores for train and test for different neighbours and decide which k should be used for final model

from sklearn.neighbors import KNeighborsClassifier


test_scores = []
train_scores = []

for i in range(1,15):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))


# In[30]:


max_train_score = max(train_scores)
max_test_score = max(test_scores)


# In[31]:


for i, v in enumerate(train_scores):
    if v == max_train_score:
        train_ind = i+1


# In[36]:


train_ind,max_train_score


# In[33]:


for i, v in enumerate(test_scores):
    if v == max_test_score:
        test_ind = i+1


# In[37]:


test_ind,max_test_score


# In[54]:


plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,15),test_scores,marker='o',label='Test Score')


# In[ ]:





# # K-means

# In[38]:


from sklearn.cluster import KMeans


# In[49]:


from sklearn.metrics import confusion_matrix,classification_report


# In[48]:



# Fit the input data. Create labels and get inertia
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(X_train)
labels = algorithm.labels_
centroids = algorithm.cluster_centers_


# In[50]:


cm = confusion_matrix(y_train,labels)


# In[51]:


print(cm)


# In[52]:


print(classification_report(y_train,labels))


# ### We can see the accuracy is very low for K-Means, so we will use KNN algorithm for final model

# # Final Model

# In[55]:


# k =14
knn = KNeighborsClassifier(14)

knn.fit(X_train,y_train)
knn.score(X_test,y_test)


# ## Confusion Matrix

# In[59]:


y_pred = knn.predict(X_test)


# In[63]:


print("Confsion Matrix")
print(confusion_matrix(y_test,y_pred))


# In[64]:


print("Classification Report")
print(classification_report(y_test,y_pred))


# ## Accuracy = 51%

# # Prediction on test data

# In[69]:


test['Segmentation'] = knn.predict(test[features])


# In[70]:


test.head()


# In[73]:


test['Segmentation'] = test['Segmentation'].apply(str)


# In[75]:


mapping ={'0':'A','1':'B','2':'C','3':'D'}


# In[77]:


test['Segmentation'] = test['Segmentation'].replace({'0':'A','1':'B','2':'C','3':'D'})


# In[78]:


test.head()


# In[79]:


submission = test[['ID','Segmentation']]


# In[80]:


submission.to_csv("Submission.csv",index = False)


# In[ ]:




