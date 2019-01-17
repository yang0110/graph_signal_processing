import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge
from utils import create_networkx_graph
from sklearn import datasets
path='../results/Bound/theoretical bound/'

np.random.seed(2019)

def lambda_(noise_level, d, user_num, dimension, item_num):
	lam=8*np.sqrt(noise_level)*np.sqrt(d)*np.sqrt(user_num+dimension)/(user_num*item_num)
	return lam 

def ridge_bound_fro(lam, rank, I_user_fro, I_min, k):
	bound=lam*(np.sqrt(rank)+2*I_user_fro)/(k+lam*I_min)
	return bound 

def ridge_bound_infty(lam, rank, I_user_infty, I_min, k):
	bound=lam*np.sqrt(rank)*(1+2*I_user_infty)/(k+lam*I_min)
	return bound 

def graph_ridge_bound_fro(lam, rank, lap_user_fro, lap_min, k):
	bound=lam*(np.sqrt(rank)+2*lap_user_fro)/(k+lam*lap_min)
	return bound 

def graph_ridge_bound_infty(lam, rank, lap_user_infty, lap_min, k):
	bound=lam*np.sqrt(rank)*(1+2*lap_user_infty)/(k+lam*lap_min)
	return bound 


def graph_ridge_decomp(user_num, dimension, X, Y, lam, lap_evalues):
	user_est=np.zeros((user_num, dimension))
	I=np.identity(dimension)
	for t in range(user_num):
		lam_t=lam*lap_evalues[t] 
		y=Y[t]
		a=np.dot(y, X)
		b=np.linalg.pinv(np.dot(X.T, X)+lam_t*I)
		user_est[t]=np.dot(a,b)
	return user_est 

def ridge_decomp(user_num, dimension, X, Y, lam):
	user_est=np.zeros((user_num, dimension))
	I=np.identity(dimension)
	for t in range(user_num):
		y=Y[t]
		a=np.dot(y, X)
		b=np.linalg.pinv(np.dot(X.T, X)+lam*I)
		user_est[t]=np.dot(a,b)
	return user_est 

user_num=50
dimension=10
item_num=500
noise_level=0.25
d=2

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
Sigma=np.cov(item_f.T)
u,s,v=np.linalg.svd(Sigma)
sigma_min=np.min(s)
k=sigma_min/18

user_f=np.random.normal(size=(user_num, dimension))
# user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
rank=np.linalg.matrix_rank(user_f)

ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
thrs_list=np.round(np.linspace((min_adj+max_adj)/2, max_adj, 5), decimals=4)
adj=ori_adj.copy()
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)
lap_evalues, lap_vectors=np.linalg.eig(lap)
lap_evalues=np.sort(lap_evalues)
lap_min=np.min(lap_evalues)
lap_2=lap_evalues[1]
lap_user_fro=np.linalg.norm(np.dot(lap, user_f), 'fro')
lap_user_infty=np.linalg.norm(np.dot(lap, user_f), np.inf)
evalues_matrix=np.diag(lap_evalues)
lam_user_fro=np.linalg.norm(np.dot(evalues_matrix, user_f), 'fro')
I=np.identity(user_num)
I_ev, I_evc=np.linalg.eig(I)
I_ev=np.sort(I_ev)
I_min=np.min(I_ev)
I_2=I_ev[1]
I_user_fro=np.linalg.norm(np.dot(I, user_f), 'fro')
I_user_infty=np.linalg.norm(np.dot(I, user_f), np.inf)

ridge_array=np.zeros(item_num)
graph_ridge_array=np.zeros(item_num)
graph_ridge_simple_array=np.zeros(item_num)
lam_list=np.zeros(item_num)
for i in range(item_num):
	lam=lambda_(noise_level, d, user_num, dimension, i+1)
	lam2=lam*0.1
	lam_list[i]=lam
	ridge_array[i]=ridge_bound_fro(lam, rank, I_user_fro, I_2, k)
	graph_ridge_array[i]=graph_ridge_bound_fro(lam2, rank, lap_user_fro, lap_2, k)
	graph_ridge_simple_array[i]=graph_ridge_bound_fro(lam2, rank, lam_user_fro, lap_2, k)

plt.figure(figsize=(5,5))
plt.plot(ridge_array, label='Ridge')
plt.plot(graph_ridge_array, label='Graph-Ridge')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Sample size (m)', fontsize=12)
plt.ylabel('Theoretical Bound', fontsize=12)
plt.savefig(path+'01_lambda_ridge_vs_graph_ridge_noise_%s'%(noise_level)+'.png', dpi=200)
plt.show()








