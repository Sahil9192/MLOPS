#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING NECESSARY LIBRARIES

# In[2]:


get_ipython().system('pip install scikit-learn pandas matplotlib notebook')
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


# ## DATA LOADING AND PREPROCESSING

# In[6]:


data = load_wine()
X = data.data
y = data.target


# In[7]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ## SPLITING THE DATA

# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# ## TRAINING THE MODEL

# In[9]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[10]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# ## PLOTING OF HISTOGRAM

# In[11]:


plt.hist(X[:, 0], bins=20)
plt.title("Feature Distribution - Alcohol")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")
plt.show()


# In[ ]:




