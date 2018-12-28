import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
#os.chdir('Documents/code/')
import datetime 
import networkx as nx
from utils import *

user_num=30
item_num=5000
dimension=25

noise_level=0.25

g_lambda=0.1


node_f=np.random.normal(size=(user_num, 2))
adj=rbf_kernel(node_f)
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)+np.identity(user_num)
cov=np.linalg.pinv(lap)
user_f=np.random.multivariate_normal(np.zeros(user_num), cov, size=dimension).T 

##generate item_f
a=np.random.normal(size=(dimension, dimension))
item_cov=np.dot(a.T, a)
item_f=np.random.multivariate_normal(np.zeros(dimension), item_cov, size=item_num)

user_f=Normalizer().fit_transform(user_f)
item_f=Normalizer().fit_transform(item_f)


## signal 
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 

## slyvetser equation
ge=np.zeros(item_num)
re=np.zeros(item_num)
for i in range(item_num):
	print('i', i)
	x=item_f[:i+1, :]
	y=noisy_signal[:, :i+1]
	A=g_lambda*lap 
	B=np.dot(x.T, x)
	C=np.dot(y, x)
	g_ridge=scipy.linalg.solve_sylvester(A, B, C)
	ridge=np.dot(np.linalg.inv(np.dot(x.T, x)+g_lambda*np.identity(dimension)), np.dot(x.T, y.T)).T
	ge[i]=np.linalg.norm(g_ridge-user_f, 'fro')
	re[i]=np.linalg.norm(ridge-user_f, 'fro')

plt.figure(figsize=(5,5))
plt.plot(ge, 'r', label='graph ridge')
plt.plot(re, 'y', label='ridge')
plt.legend(loc=0)
plt.show()
