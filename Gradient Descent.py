#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

df = pd.read_csv("insurance_dataa.csv")


# In[2]:


df.head()


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age', "affordibility"]], df.bought_insurance, test_size=0.2, random_state = 25)


# In[4]:


len(X_train)


# In[5]:


X_train_scaled = X_train.copy()
X_train_scaled["age"] = X_train_scaled["age"]/100

X_test_scaled = X_test.copy()
X_test_scaled["age"] = X_test_scaled["age"]/100


# In[6]:


X_train_scaled


# In[8]:


model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation = "sigmoid", kernel_initializer="ones", bias_initializer = "zeros")
])

model.compile(optimizer= "SGD",
             loss = "binary_crossentropy",
             metrics=["accuracy"])

model.fit(X_train_scaled, y_train, epochs = 5000)


# In[22]:


model.evaluate(X_test_scaled, y_test)


# In[23]:


model.predict(X_test_scaled)


# In[24]:


y_test


# In[25]:


coef, intercept = model.get_weights()
coef, intercept


# In[ ]:




