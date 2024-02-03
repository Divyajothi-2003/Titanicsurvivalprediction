#!/usr/bin/env python
# coding: utf-8

# In[38]:


from sklearn.tree import DecisionTreeClassifier
import pandas as pd
df = pd.read_csv('titanic.csv')
print(df.to_string())


# In[39]:


print(df.shape)
print(df.isna().sum())


# In[40]:


mean=df['Age'].mean()
print(mean)


# In[41]:


df['Age']=df['Age'].fillna(mean)
df


# In[42]:


newdf = df.drop(columns = ['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'])
print(newdf.to_string())


# In[43]:


newdf1 = pd.get_dummies(newdf,dtype=int)
print(newdf1.to_string())


# In[44]:


x = newdf1.drop(columns = ['Survived'])
print(x)
y = df['Survived']
print(y)


# In[45]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.20)


# In[46]:


model = DecisionTreeClassifier()
model.fit(X_train,Y_train)


# In[47]:


print(X_train.shape)
print(X_test.shape)


# In[48]:


model.predict(X_test)


# In[49]:


model.score(X_train,Y_train)


# In[ ]:




