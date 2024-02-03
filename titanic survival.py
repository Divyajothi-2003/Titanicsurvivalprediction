#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# In[22]:


df=pd.read_csv('titanic.csv')
print(df.to_string())


# In[23]:


imputer = SimpleImputer(strategy='mean')
X.loc[:, 'Age'] = imputer.fit_transform(X[['Age']])


# In[24]:


from sklearn.model_selection import train_test_split
X_train_1, X_test_1, Y_train_1, Y_test_1 = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[25]:



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X.loc[:, 'Age'] = imputer.fit_transform(X[['Age']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the transformers for numerical and categorical columns
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = StandardScaler()

categorical_features = ['Sex', 'Embarked']
categorical_transformer = OneHotEncoder(drop='first', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing and train the model
preprocessor.fit(X_train)
X_train_1 = preprocessor.transform(X_train)
X_test_1 = preprocessor.transform(X_test)

model = LinearRegression()
model.fit(X_train_1, y_train)

score = model.score(X_test_1, y_test)
print(f'Model R^2 Score: {score:.2f}')


# In[26]:


numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
numeric_transformer = StandardScaler()


# In[27]:


categorical_features = ['Sex', 'Embarked']
categorical_transformer = OneHotEncoder(drop='first', sparse=False)


# In[28]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[ ]:




