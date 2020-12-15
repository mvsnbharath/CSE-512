#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data = sio.loadmat('mnist.mat')
for k in range(9):
    plt.subplot(3,3,k+1)
    plt.imshow(np.reshape(data['trainX'][k,:],(28,28)))
    plt.title(data['trainY'][0,k])
    plt.tight_layout()


# In[3]:


X = data['trainX']
y = data['trainY']
xTest = data['testX']
yTest = data['testY']

idx = np.logical_or(np.equal(y,4) , np.equal(y,9))
idxTest = np.logical_or(np.equal(yTest,4) , np.equal(yTest,9))
X = X[idx[0, :],  :]
y = y[idx]
xTest = xTest[idxTest[0, :],  :]
yTest = yTest[idxTest]

y[np.equal(y,4)] = 0
y[np.equal(y,9)] = 1

yTest[np.equal(yTest,4)] = 0
yTest[np.equal(yTest,9)] = 1

print("XTrain Shape modified: " +str(np.shape(X)))
print("yTrain Shape modified: " +str(np.shape(y)))
print("XTest Shape modified: " +str(np.shape(xTest)))
print("yTest Shape modified: " +str(np.shape(yTest)))


# In[16]:


## Helper Functions

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def normalizeData(X):
    # finding mean
    mean=np.mean(X,axis=0)
    # divide with (max -min)
    # maxValue = 255, minValue = 0
    X_norm = (X - mean)/255
    return X_norm 

def costFunction(weights,X, y): 
    m=len(y)
    z = np.dot(X, weights)
    loss_for_1 = y * np.log(sigmoid(z))
    loss_for_0 = (1 - y) * np.log(1 - sigmoid(z))
    grad = 1/m * np.dot(X.transpose(),(sigmoid(z) - y))
    return -sum(loss_for_1 + loss_for_0) / m, grad

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    lossFunctionHistory =[]
    
    for i in range(num_iters):
        loss, gradient = costFunction(theta,X,y)
        theta = theta - (alpha * gradient)
        lossFunctionHistory.append(loss)    
    return theta , lossFunctionHistory


def predict(theta, X):  
    p = np.round(sigmoid(X.dot(theta)))
    return p


# In[17]:


m , n = X.shape[0], X.shape[1]
X_normalized = normalizeData(X)
X_normalized = np.append(np.ones((m,1)),X_normalized,axis=1)
y = y.reshape(m,1)
initial_theta = np.zeros((n+1,1))
cost, grad= costFunction(initial_theta,X_normalized,y)
print("Initial theta cost is",cost)


# In[18]:


theta , J_history = gradientDescent(X_normalized,y,initial_theta,0.001,5000)


# In[19]:


plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")


# In[20]:


print("Final Loss is ",J_history[-1])


# In[21]:


m , n = xTest.shape[0], xTest.shape[1]
Xtest_normalized = normalizeData(xTest)
Xtest_normalized = np.append(np.ones((m,1)),Xtest_normalized,axis=1)
p = predict(theta, Xtest_normalized)
print('Train Accuracy: {:.2f} %'.format(np.mean(p == yTest) * 100))


# In[ ]:




