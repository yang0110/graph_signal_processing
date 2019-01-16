import numpy as np 
import cvxpy as cp 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from utils import *
from sklearn import datasets
path='../results/Graph_ridge_results/'

user_num=100
dimension=10
item_num=100

user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)
adj=rbf_kernel(user_f)
degree=np.diag(np.sum(adj, axis=1))-adj
degree=np.diagonal(degree)
degree=np.sort(degree)
lap=csgraph.laplacian(adj, normed=False)
lap_evalues, lap_evectors=np.linalg.eig(lap)
lap_evalues=np.sort(lap_evalues)
fig, (ax1, ax2)=plt.subplots(1,2)
ax1.plot(lap_evalues[1:], '+-', label='eigen values')
ax1.legend(loc=0, fontsize=10)
ax2.plot(degree[1:],'.-', label='degree')
ax2.legend(loc=0, fontsize=12)
plt.show()