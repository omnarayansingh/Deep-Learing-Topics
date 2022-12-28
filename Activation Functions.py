#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math

def sigmoid(x):
    return 1/ (1 + math.exp(-x))


# In[10]:


sigmoid(-200)


# In[11]:


def tanh(x):
    return(math.exp(x) - math.exp(-x) /  math.exp(x) + math.exp(-x))


# In[13]:


tanh(-4)


# In[14]:


def relu(x):
    return max(0,x)


# In[16]:


relu(7)


# In[17]:


relu(-2)


# In[19]:


def leaky_relu(x):
    return max (0.1*x, x)


# In[21]:


leaky_relu(-10)


# In[ ]:




