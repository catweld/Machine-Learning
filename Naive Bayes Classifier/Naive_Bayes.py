#!/usr/bin/env python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
# 
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
# 
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
# 
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.
# 
# Be sure to put `%matplotlib inline` at the top of every code cell where you call plotting functions to get the resulting plots inside the document.

# ## Import the libraries
# 
# In Jupyter, select the cell below and press `ctrl + enter` to import the needed libraries.
# Check out `labfuns.py` if you are interested in the details.

# In[639]:


import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random
import math


# ## Bayes classifier functions to implement
# 
# The lab descriptions state what each function should do.

# In[661]:


def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    
    counts = np.zeros(Nclasses)
    i = 0
    for c in classes:
        idx = (labels == c)
        counts[i] = np.sum(W[idx])
        i += 1
    return counts/np.sum(counts)

def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts) # Get the x for the class labels. Vectors are rows.pts)
        
    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    for jdx, c in enumerate(classes): 
        idx = labels==c 
        xlc = X[idx,:]
        w = W[idx]
        mu[jdx] = np.sum((xlc*w), axis=0)/np.sum(w)
        sigma[jdx] = np.diag(np.sum(w*np.square(xlc - mu[jdx]),axis=0)/np.sum(w))                                          
    return mu, sigma

def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))
    C = 1
    for jdx in range(Nclasses):
        sigmaInv = np.linalg.inv(sigma[jdx])
        muDiff = X-mu[jdx]
        logProb[jdx] = -1/2*np.log(np.linalg.det(sigma[jdx]))-1/2*np.diag(np.dot(np.dot(muDiff,sigmaInv),np.transpose(muDiff)))+np.log(prior[jdx])
    h = np.argmax(logProb,axis=0)
    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:

# In[662]:


# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.

# In[657]:


get_ipython().run_line_magic('matplotlib', 'inline')

X, labels = genBlobs(centers=5)
mu, sigma = mlParams(X,labels)
plotGaussian(X,labels,mu,sigma)


# Call the `testClassifier` and `plotBoundary` functions for this part.

# In[635]:


testClassifier(BayesClassifier(), dataset='iris', split=0.7)


# In[626]:


testClassifier(BayesClassifier(), dataset='vowel', split=0.7)


# In[627]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotBoundary(BayesClassifier(), dataset='iris',split=0.7)


# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.

# In[730]:


def trainBoost(base_classifier, X, labels, T=10):
    Npts,Ndims = np.shape(X)

    classifiers = [] 
    alphas = [] 
    wCur = np.ones((Npts,1))/float(Npts)
    
    for i_iter in range(0, T):
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))
        vote = classifiers[-1].classify(X)
        h = np.zeros(Npts)
        for x in range(Npts):
            if vote[x] == labels[x]:
                h[x] = 1
            else:
                h[x] = 0
        E = np.dot(np.transpose(wCur), np.transpose(np.array([(np.ones(Npts) - h)])))
        alpha = 1/2*(math.log(1 - E + .00001) - math.log(E + .00001))
        
        weight = math.exp(-alpha)*(vote == labels) + math.exp(alpha)*(vote != labels)
        
        wCur = np.multiply(wCur, np.array([weight]).T)
        wCur = wCur/np.sum(wCur)
        alphas.append(alpha)
        
    return classifiers, alphas

def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))
        for classifier, alpha in zip(classifiers, alphas):
            for idx, c in enumerate(classifier.classify(X)):
                votes[idx][c] += alpha

        return np.argmax(votes,axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class. This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument. No need to add anything here.

# In[731]:


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.

# In[748]:


testClassifier(BoostClassifier(BayesClassifier(), T=4), dataset='iris',split=0.7)


# In[733]:


testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)


# In[740]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotBoundary(BoostClassifier(BayesClassifier(),T=10), dataset='iris',split=0.7)


# Now repeat the steps with a decision tree classifier.

# In[749]:


testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)


# In[750]:


testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# In[751]:


testClassifier(DecisionTreeClassifier(), dataset='vowel',split=0.7)


# In[752]:


testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)


# In[753]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotBoundary(DecisionTreeClassifier(), dataset='iris',split=0.7)


# In[754]:


get_ipython().run_line_magic('matplotlib', 'inline')
plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)


# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.

# In[ ]:


testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)


# In[ ]:


testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging to one of 40 persons!

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
X,y,pcadim = fetchDataset('olivetti') 
xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) 
pca = decomposition.PCA(n_components=20)
pca.fit(xTr) 
xTrpca = pca.transform(xTr)
xTepca = pca.transform(xTe)
classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
yPr = classifier.classify(xTepca)
testind = random.randint(0, xTe.shape[0]-1)
visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])

