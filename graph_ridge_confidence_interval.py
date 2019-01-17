import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('C:/Kaige_Research/Code/GSP/code/')
import datetime 
import networkx as nx
from sklearn import datasets
path='../results/graph_ridge_and_approximation/'

def graph_ridge_confidence(R, S, lam, x, delta, evalue, I):
	V=lam*evalue*I
	V_t=np.dot(x.T, x)+V 
	a=np.sqrt(np.linalg.det(V_t))/np.sqrt(np.linalg.det(V))
	b=np.sqrt(2*np.log(a/delta))
	c=R*b+np.sqrt(lam)*S
	return c 

def ridge_confidence(R, S, lam, x, delta, I):
	V=lam*I
	V_t=np.dot(x.T, x)+V 
	a=np.sqrt(np.linalg.det(V_t))/np.sqrt(np.linalg.det(V))
	b=np.sqrt(2*np.log(a/delta))
	c=R*b+np.sqrt(lam)*S
	return c 

user_num=50
item_num=100
dimension=10
cluster_std=10
noise_level=0.1
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

R=noise_level 
delta=0.05
S=1
lam=0.1
I=np.identity(dimension)

ridge_conf=np.zeros(user_num)
graph_ridge_conf=np.zeros(user_num)
for i in range(user_num):
	x=item_f[:i+1]
	ridge_conf[i]=ridge_confidence(R,S, lam, x, delta, I)
	graph_ridge_conf[i]=graph_ridge_confidence(R, S, lam, x, delta, lap_evalues[1], I)

plt.figure()
plt.plot(ridge_conf, label='ridge')
plt.plot(graph_ridge_conf, label='graph-ridge')
plt.legend(loc=0, fontsize=12)
plt.show()
