import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge, Graph_ridge_simple
from sklearn import datasets
from utils import *
path='../results/graph_ridge_and_approximation/'

user_num=50
item_num=100
dimension=10
cluster_std=0.1
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

user_f=np.random.normal(size=(user_num, dimension))
user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=cluster_std, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
adj=rbf_kernel(user_f)
lap=csgraph.laplacian(adj, normed=False)
lap_evalues, lap_evectors=np.linalg.eig(lap)
idx=np.argsort(lap_evalues)
lap_evalues=lap_evalues[idx]
lap_evectors=lap_evectors.T[idx].T
evalues_matrix=np.diag(lap_evalues)


plt.figure(figsize=(5,5))
plt.pcolor(adj)
plt.colorbar()
plt.savefig(path+'adj_user-num_cluster_std_%s_%s'%(user_num, cluster_std)+'.png', dpi=100)
plt.show()

a=0
b=0
lap_norm=np.zeros(user_num)
lam_norm=np.zeros(user_num)
for i in range(user_num):
	a+=lap_evalues[i]*(np.linalg.norm(np.dot(lap_evectors[:,i], user_f))**2)
	b+=lap_evalues[i]*(np.linalg.norm(user_f[i])**2)
	lap_norm[i]=a
	lam_norm[i]=b

plt.figure(figsize=(5,5))
plt.plot(np.diff(lap_norm), label='Graph-Ridge')
plt.plot(np.diff(lam_norm), label='Graph-Ridge Approx')
plt.xlabel('k=[1,n]', fontsize=12)
plt.ylabel('Spectrum', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.savefig(path+'spectum_user_num_cluster_std_%s_%s'%(user_num, cluster_std)+'.png', dpi=100)
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.plot(lap_norm, label='Graph-Ridge')
plt.plot(lam_norm, label='Graph-Ridge Approx')
plt.xlabel('k=[1, n]', fontsize=12)
plt.ylabel('Cumulative Spectrum', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.savefig(path+'spectum_user_num_cluster_std_%s_%s'%(user_num, cluster_std)+'.png', dpi=100)
plt.tight_layout()
plt.show()



