#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('Iris.csv')


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# # Exploratory Data Analytics 

# In[7]:


df.isnull().sum()


# In[8]:


df.corr()


# In[9]:


plt.figure(figsize=(9,9))
sns.heatmap(df.corr(),annot=True)


# In[10]:


sns.pairplot(df)


# In[11]:


numerical_features=[feature for feature in df.columns if df[feature].dtype!='O']


# In[12]:


for feature in numerical_features:
    print(df[feature].value_counts())


# In[13]:


continuous_features=[feature for feature in df.columns if len(df[feature].unique())>25]


# In[14]:


continuous_features


# In[15]:


for feature in continuous_features:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.show()


# #  Feature Engineering 

# In[16]:


df=df.drop(['Id'],axis=1)


# In[17]:


df


# In[18]:


df['Species'].value_counts()


# In[19]:


dict1={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}


# In[20]:


df['target']=df['Species'].map(dict1)


# In[21]:


df


# In[22]:


df=df.drop(['Species'],axis=1)


# In[23]:


df


# In[24]:


X=df.iloc[:,:-1]


# In[25]:


X


# In[26]:


y=df.iloc[:,-1]


# In[27]:


y


# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# In[29]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# # Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression


# In[31]:


lr=LogisticRegression()


# In[32]:


lr.fit(X_train,y_train)


# In[33]:


lr.score(X_train,y_train)


# In[34]:


lr.score(X_test,y_test)


# In[35]:


y_pred=lr.predict(X_test)


# In[36]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[37]:


accuracy_score(y_pred,y_test)


# In[38]:


print(classification_report(y_pred,y_test))


# In[39]:


print(confusion_matrix(y_pred,y_test))


# # K-NN

# In[40]:


from sklearn.neighbors import KNeighborsClassifier


# In[41]:


knn=KNeighborsClassifier(n_neighbors=5)


# In[42]:


knn.fit(X_train,y_train)


# In[43]:


knn.score(X_train,y_train) 


# In[44]:


knn.score(X_test,y_test)


# In[45]:


y_pred1=knn.predict(X_test)


# In[46]:


print(accuracy_score(y_pred,y_test))


# In[47]:


print(classification_report(y_pred,y_test))


# In[48]:


print(confusion_matrix(y_pred,y_test)) 


# In[ ]:




