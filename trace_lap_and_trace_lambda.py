import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
# os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from sklearn import datasets
from utils import *
path='../results/'

user_num=50
dimension=5
cluster_std=10
I=np.identity(user_num)
user_f=np.random.normal(size=(user_num, dimension))
user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=cluster_std, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
adj=rbf_kernel(user_f)
lap=csgraph.laplacian(adj, normed=False)
lap_evalues, lap_evectors=np.linalg.eig(lap)
idx=np.argsort(lap_evalues)
lap_evalues=lap_evalues[idx]
lap_evectors=lap_evectors.T[idx].T
Lambda=np.diag(lap_evalues)

plt.figure()
plt.pcolor(adj)
plt.colorbar()
plt.title('cluster std=%s'%(cluster_std), fontsize=12)
plt.savefig(path+'adj_pcolor_cluster_std_%s'%(cluster_std)+'.png', dpi=100)
plt.show()
 
en_lap=np.zeros(user_num)
en_lam=np.zeros(user_num)
for i in range(user_num):
	en_lap[i]=lap_evalues[i]*(np.linalg.norm(np.dot(lap_evectors[:,i].reshape((1, user_num)), user_f))**2)
	en_lam[i]=lap_evalues[i]*(np.linalg.norm(user_f[i])**2)

cum_lap=np.cumsum(en_lap)
cum_lam=np.cumsum(en_lam)

fig, (ax1, ax2)=plt.subplots(1,2)
ax1.plot(en_lap, label='lap')
ax1.plot(en_lam, label='lam')
ax1.legend(loc=0)
ax1.set_title('energy', fontsize=12)
ax2.plot(cum_lap, label='lap')
ax2.plot(cum_lam, label='lam')
ax2.legend(loc=0)
ax2.set_title('cum energy \n cluster_std=%s'%(cluster_std), fontsize=12)
plt.savefig(path+'trace_lap_vs_trace_lam_cluster_std_%s'%(cluster_std)+'.png', dpi=100)
plt.show()

print('sum_lap', cum_lap[-1])
print('sum_lam', cum_lam[-1])





