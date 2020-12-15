#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn import metrics
import math
from numpy import linalg as LA


# In[2]:


q4Data = sio.loadmat('weatherDewTmp.mat')


# In[3]:


plt.plot(q4Data['weeks'].T, q4Data['dew'].T)
plt.xlabel('Weeks after first reading')
plt.ylabel('Dew point temp (C)')
plt.show()


# In[4]:


X = q4Data['weeks'].T
y = q4Data['dew'].T


# In[5]:


def conditionNumber(alpha,p,X):
    poly = PolynomialFeatures(degree= p-1)
    modifiedX = poly.fit_transform(X)
    X_reg = np.dot(modifiedX.T,modifiedX) + alpha*np.identity(p)
    return LA.cond(X_reg)


# In[6]:


X = q4Data['weeks'].T
m = np.shape(X)[0]
pValues = [2,3,6,11]
alphaValues = [0,0.1*m,m,10*m,100*m]
for p in pValues:
    for alpha in alphaValues:
        A = conditionNumber(alpha, p, X)
        print("p = "+str(p), " alpha = " +str(alpha)+ " condition number for A =  "+ str(A))


# ## 4f
# 
# 
# |p| 	$$ A,  \alpha = 0$$|$$ A_{reg}, \alpha = 0.1 *m$$|$$ A_{reg}, \alpha = m$$|$$ A_{reg}, \alpha = 10*m$$|$$ A_{reg}, \alpha = 100*m$$|
# |--|-------------------------------|-----|-----|-----|-----|
# |2 |32.47|22.75|6.75|1.68|1.07|
# |3 |1476.39|530.04|79.11|9.20|1.82|
# |6 |730851302.28|2707863.72|271693.24|27179.31|2718.92|
# |11 |7.299e+18|3970315661934.136|397031763152.4492|39703178289.79932|3970317849.5246153|

# In[7]:


def plotPolyFitLinearRegression(x,y,pValues):
    plt.figure(figsize=(20,10))
    for k in range(len(pValues)):
        plt.subplot(3,2,k+1)
        polynomial_features= PolynomialFeatures(degree=pValues[k-1]-1)
        x_poly = polynomial_features.fit_transform(x)
        model = LinearRegression()
        model.fit(x_poly, y)
        plt.title("p = "+str(pValues[k-1]))
        y_poly_pred = model.predict(x_poly)
        plt.plot(x, y)
        plt.plot(x, y_poly_pred,'r')
        plt.xlabel('Weeks after first reading')
        plt.ylabel('Dew point temp (C)')
    plt.tight_layout()
    return


# ## 4g

# In[8]:


x = q4Data['weeks'].T
y = q4Data['dew'].T
pValues = [ 2,10, 100,150, 200,1]
plotPolyFitLinearRegression(x,y,pValues)


# In[9]:


def plotPolyFitRidge(x,y,pValues,alpha=0.0001):
    plt.figure(figsize=(20,10))
    for k in range(len(pValues)):
        plt.subplot(3,2,k+1)
        plt.title("p = "+str(pValues[k-1]))
        polynomial_features= PolynomialFeatures(degree=pValues[k-1]-1)
        x_poly = polynomial_features.fit_transform(x)
        model = Ridge(alpha)
        model.fit(x_poly, y)
        y_poly_pred = model.predict(x_poly)
        plt.plot(x, y)
        plt.plot(x, y_poly_pred,'r')
        plt.xlabel('Weeks after first reading')
        plt.ylabel('Dew point temp (C)')
    plt.tight_layout()
    return


# ## 4h

# In[10]:


x = q4Data['weeks'].T
y = q4Data['dew'].T
pValues = [ 2,10, 100,150, 200,1]
plotPolyFitRidge(x,y,pValues)


# ## 4i

# In[11]:


# adding one more week
xNew = []
temp = X[-1]
stepSize = 0.00595238
for i in range(1,150):
    xNew.append(temp + (i*stepSize))


# In[12]:


xFinal = np.concatenate((x, xNew))


# In[13]:


polynomial_features= PolynomialFeatures(degree=29)
x_poly = polynomial_features.fit_transform(x)
model = Ridge(alpha)
model.fit(x_poly, y)
x_poly_1 = polynomial_features.fit_transform(xNew)
y_poly_pred = model.predict(x_poly_1)


# In[14]:


yFinal = np.concatenate((y, y_poly_pred))


# In[15]:


plt.figure(figsize=(20,10))
plt.title("p = "+str(30))
# plt.plot(x, y)
plt.plot(xNew, y_poly_pred,'r')
plt.xlabel('Weeks after first reading')
plt.ylabel('Dew point temp (C)')
plt.show()


# In[ ]:




