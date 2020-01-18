#!/usr/bin/env python
# coding: utf-8

# In[119]:


import sys
import math
import dtree
import random
import numpy
import monkdata as m
import matplotlib.pyplot as pyplot
import drawtree_qt5 as graph
import statistics
monktest1 = open("monks-1.test")
monktest2 = open("monks-2.test")
monktest3 = open("monks-3.test")
monktrain1 = open("monks-1.train")
monktrain2 = open("monks-2.train")
monktrain3 = open("monks-3.train")
from PyQt5 import QtCore, QtGui, QtWidgets


# In[2]:


#calculate entropy of data set
entropym1 = dtree.entropy(m.monk1)
print(entropym1)
entropym2 = dtree.entropy(m.monk2)
print(entropym2)
entropym3 = dtree.entropy(m.monk3)
print(entropym3)


# In[3]:


#give example of uniform
uniform = (
m.Sample(True, (random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0)), 1000),
m.Sample(True, (random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0)), 1001),
m.Sample(False, (random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0)), 1002),
m.Sample(False, (random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0)), 1003))


# In[4]:


nonuniform = (
m.Sample(True, (random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0)), 1000),
m.Sample(True, (random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0)), 1001),
m.Sample(True, (random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0)), 1002),
m.Sample(True, (random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0), random.uniform(1,0)), 1003))


# In[5]:


entropyuniform = dtree.entropy(uniform)
entropynonuniform = dtree.entropy(nonuniform)


# In[6]:


#calculate average gain
for i in range(6):
    print(dtree.averageGain(m.monk1,m.attributes[i]))
#monk 1 is highest information gain at a5
for i in range(6):
    print(dtree.averageGain(m.monk2,m.attributes[i]))
#monk 2 is highest information gain at a5
for i in range(6):
    print(dtree.averageGain(m.monk3,m.attributes[i]))    
#monk 3 is highest information gain at a2


# In[7]:


#split monk1 data by highest average gain
monk1bestattribute = dtree.bestAttribute(m.monk1,m.attributes)
monk11 = dtree.select(m.monk1,monk1bestattribute,1)
monk12 = dtree.select(m.monk1,monk1bestattribute,2)
monk13 = dtree.select(m.monk1,monk1bestattribute,3)
monk14 = dtree.select(m.monk1,monk1bestattribute,4)


# In[8]:


entropy11 = dtree.entropy(monk11)
entropy12 = dtree.entropy(monk12)
entropy13 = dtree.entropy(monk13)
entropy14 = dtree.entropy(monk14)


# In[10]:


monk1bestattribute = dtree.bestAttribute(m.monk1,m.attributes)
for i in monk1bestattribute.values:
    monksplit = dtree.select(m.monk1,monk1bestattribute,i)
    monk1bestattribute2 = dtree.bestAttribute(monksplit,m.attributes)
    for j in range(6):


# In[11]:


monk1bestattribute = dtree.bestAttribute(m.monk1,m.attributes)
for i in monk1bestattribute.values:
    monksplit = dtree.select(m.monk1,monk1bestattribute,i)
    monk1bestattribute2 = dtree.bestAttribute(monksplit,m.attributes)
    print(i)
    for j in monk1bestattribute2.values:
        monksplit2 = dtree.select(monksplit,monk1bestattribute2,j)
        print(dtree.mostCommon(monksplit2))


# In[12]:


tree1 = dtree.buildTree(m.monk1, m.attributes)
#graph.drawTree(tree1)


# In[13]:


tree2 = dtree.buildTree(m.monk2, m.attributes)
#graph.drawTree(tree2)


# In[14]:


tree3 = dtree.buildTree(m.monk3, m.attributes)
#graph.drawTree(tree3)


# In[89]:


def partition(data, fraction): 
    ldata = list(data)
    random.shuffle(ldata) 
    breakPoint = int(len(ldata) * fraction) 
    return ldata[:breakPoint], ldata[breakPoint:]


# In[174]:


def prune(monktrain,monkval):
    pruned = []
    trees = dtree.allPruned(dtree.buildTree(monktrain, m.attributes))
    for prunedtree in trees: 
        pruned.append(dtree.check(prunedtree, monkval))
    tree = trees[numpy.argmax(pruned)]
    iteration = 1
    error = 1-dtree.check(prunedtree, monkval)
    if error<=1-pruned[numpy.argmax(pruned)]:
        while error<=1-pruned[numpy.argmax(pruned)]:
            trees = dtree.allPruned(tree)
            pruned = []
            for prunedtree in trees: 
                pruned.append(dtree.check(prunedtree, monkval))
            tree = trees[numpy.argmax(pruned)]
            iteration = iteration + 1
            return(tree)
    else:
        return(tree)


# In[175]:


monk1train, monk1val = partition(m.monk1, 0.6)
monk1prune = prune(monk1train,monk1val)
errormonk1 = (dtree.check(monk1prune, m.monk1test))


# In[176]:


monk3train, monk3val = partition(m.monk3, 0.6)
monk3prune = prune(monk3train,monk3val)
errormonk3 = (dtree.check(monk3prune, m.monk3test))
print(errormonk3)


# In[177]:


fractionarray = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
classerror1 = []
x1 = []
for i in fractionarray:
    for j in range(100): 
        monk1train, monk1val = partition(m.monk1,i)
        monk1prune = prune(monk1train, monk1val)
        classerror1.append(1-dtree.check(monk1prune, m.monk1test))
        x1.append(i)
mean1 = statistics.mean(classerror1)
variance1 = statistics.variance(classerror1)
pyplot.scatter(x1,classerror1)
pyplot.title("Monk1 Fraction of Partition vs. Classification Error")
pyplot.legend(['Mean:' + str(mean1)+"\n"'Variance:' + str(variance1)]);
pyplot.xlabel('Partition Fraction')
pyplot.ylabel('Classification Error')


# In[178]:


classerror3 = []
x3 = []
for i in fractionarray:
    for j in range(100): 
        monk3train, monk3val = partition(m.monk3,i)
        monk3prune = prune(monk3train, monk3val)
        classerror3.append(1-dtree.check(monk3prune, m.monk3test))
        x3.append(i)
mean3 = statistics.mean(classerror3)
variance3 = statistics.variance(classerror3)
pyplot.scatter(x3,classerror3)
pyplot.title("Monk3 Fraction of Partition vs. Classification Error")
pyplot.legend(['Mean:' + str(mean3)+"\n"'Variance:' + str(variance3)]);
pyplot.xlabel('Partition Fraction')
pyplot.ylabel('Classification Error')


# In[ ]:




