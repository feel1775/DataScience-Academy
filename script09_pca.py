# libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# read data
df = pd.read_csv('data01_iris.csv')
xorg = df[ ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'] ]
xorg = xorg.as_matrix()

# PCA 
from sklearn.decomposition import PCA
pca = PCA()
xtrans = pca.fit_transform(xorg)
xorg.var(axis=0)
xorg.var(axis=0)/sum(xorg.var(axis=0))
pca.explained_variance_
pca.explained_variance_ratio_

from sklearn.preprocessing import scale
xscaled = scale(xorg,axis=0,with_mean=True,with_std=True)
xtrans = pca.fit_transform(xscaled)
xscaled.var(axis=0)
xscaled.var(axis=0)/sum(xscaled.var(axis=0))
pca.explained_variance_
pca.explained_variance_ratio_

xrecon = pca.inverse_transform(xtrans)
e = (xscaled-xrecon)**2
e.mean()

pca1 = PCA(n_components=1)
xtrans1 = pca1.fit_transform(xscaled)
xrecon = pca1.inverse_transform(xtrans1)
e = (xscaled-xrecon)**2
e.mean()


# practice

























# PLEASE DO NOT GO DOWN BEFORE YOU TRY BY YOURSELF

###########################################################
# Practice Reference Code
###########################################################

# practice
df = pd.read_csv('data03_nci_data.csv')
xorg = df.as_matrix()

from sklearn.decomposition import PCA
pca = PCA()
xtrans = pca.fit_transform(xorg)

pca.explained_variance_ratio_
pca.explained_variance_ratio_.cumsum()

pca2 = PCA(n_components=2)
xtrans = pca2.fit_transform(xorg)
xrecon = pca2.inverse_transform(xtrans)
err = np.sqrt(((xorg-xrecon)**2).mean())


