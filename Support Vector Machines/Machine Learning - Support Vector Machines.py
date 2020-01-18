#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy, random, math 
from scipy.optimize import minimize 
import matplotlib.pyplot as plt
import numpy


# In[6]:


def linKernel(x,y):
    scalarProduct = numpy.dot(numpy.transpose(x),y)
    return scalarProduct


# In[7]:


def polyKernel(x,y,p):
    result = (numpy.dot(numpy.transpose(x),y)+1)**p
    return result


# In[8]:


def RBFKernel(x,y,sigma):
    result = numpy.exp((-(numpy.linalg.norm(x-y)**2))/((2*sigma)**2))
    return result


# In[9]:


def clusterKernel(V):
    return V


# In[10]:


def objective(alpha):
    obj = (1/2)*numpy.dot(numpy.dot(alpha,P),alpha) - numpy.sum(alpha)
    return obj


# In[11]:


def zerofun(alpha):
    return numpy.dot(alpha,targets)


# In[15]:


def extractNonZeros(alpha):
    nonZeroAlphas = []
    for i in range(len(alpha)):
        if alpha[i] > .000005:
            nonZeroAlphas.append([alpha[i],inputs[i],targets[i]])
    return nonZeroAlphas


# In[16]:


def bcomp(nonZeroAlphas,C):
    b = 0
    aval = 0
    xval = 0
    tval = 0
    for alphai, inputi, targeti in nonZeroAlphas:
        if alphai < C:
            aval = alphai
            xval = inputi
            tval = targeti
            break     
    for alphai, inputi, targeti in nonZeroAlphas:
        b += alphai*targeti*linKernel(xval,inputi)
    b = b-tval
    return b, xval


# In[17]:


def indicator(xval):
    inds = 0
    for alpha, inputs, targets in nonZeroAlphas:
        inds += alpha*targets*linKernel(xval,inputs)
    inds = inds - b
    return inds


# In[ ]:


classA = numpy.concatenate((numpy.random.randn(10, 2)*0.2 + [1.5, 0.5], numpy.random.randn(10, 2)*0.2 + [-1.5, 0.5])) 
classB = numpy.random.randn(20, 2)*0.2 + [0 , -0.5]
inputs = numpy.concatenate((classA, classB)) 
targets = numpy.concatenate( (numpy.ones(classA.shape[0]), -numpy.ones(classB.shape[0]))) 
N = inputs.shape[0]
permute = list(range(N)) 
random.shuffle(permute) 
inputs = inputs[permute, :] 
targets = targets[permute]


# In[ ]:


plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.') 
plt.plot([p[0] for p in classB], [p[1] for p in classB ], 'r.') 
plt.axis('equal')
plt.savefig('svmplot.pdf') 
plt.show() 


# In[18]:


global P
P = numpy.zeros(shape=(N,N))
for i in range(N):
    for j in range(N):
        P[i][j] = targets[i]*targets[j]*linKernel(inputs[i], inputs[j])


# In[19]:


start = numpy.zeros(N)
constraint={'type':'eq', 'fun':zerofun}
C = 100
bounds=[(0, C) for b in range(N)]
alpha = start


# In[20]:


ret = minimize( objective , start , bounds = bounds, constraints = constraint)
alpha = ret.x


# In[21]:


nonZeroAlphas = extractNonZeros(alpha)
xval = numpy.zeros(2)


# In[23]:


b = bcomp(nonZeroAlphas,C)[0]
xval = bcomp(nonZeroAlphas,C)[1]


# In[24]:


inds = indicator(xval)


# In[25]:


plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.') 
plt.plot([p[0] for p in classB], [p[1] for p in classB ], 'r.') 
plt.axis('equal')
xgrid=numpy. linspace(-5, 5) 
ygrid=numpy. linspace(-4, 4) 
grid=numpy.array ([[ indicator(numpy.array((x,y))) 
                    for x in xgrid ] 
                   for y in ygrid ]) 
plt .contour(xgrid , ygrid , grid , 
             (-1.0, 0.0, 1.0) , 
             colors=('red' , 'black' , 'blue'), 
             linewidths=(1, 3, 1))

