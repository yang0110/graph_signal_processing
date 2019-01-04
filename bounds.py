import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge
from utils import create_networkx_graph
from sklearn import datasets
path='../results/Bound/'

np.random.seed(2019)

def lambda_(noise_list, d, user_num, dimension, item_num):
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


user_num=200
dimension=25
item_num=1000
noise_level=0.1
d=3

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
Sigma=np.cov(item_f.T)
u,s,v=np.linalg.svd(Sigma)
sigma_min=np.min(s)
k=sigma_min/18

user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)
rank=np.linalg.matrix_rank(user_f)
user_infty=np.linalg.norm(user_f, np.inf)
user_fro=np.linalg.norm(user_f, 'fro')

ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
thrs_list=np.round(np.linspace((min_adj+max_adj)/2, max_adj, 5), decimals=4)
adj=ori_adj.copy()
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)+np.identity(user_num)
lap_user_fro=np.linalg.norm(np.dot(lap, user_f), 'fro')
lap_user_infty=np.linalg.norm(np.dot(lap, user_f), np.inf)

I=np.identity(user_num)
I_ev, I_evc=np.linalg.eig(I)
I_min=np.max(I_ev)
I_dim=I_ev[dimension]
I_user_fro=np.linalg.norm(np.dot(I, user_f), 'fro')
I_user_infty=np.linalg.norm(np.dot(I, user_f), np.inf)
I_infty=np.linalg.norm(I, np.inf)
I_fro=np.linalg.norm(I, 'fro')

lap_evalues, lap_evectors=np.linalg.eig(lap)
lap_evalues=np.sort(lap_evalues)
lap_min=np.min(lap_evalues)
lap_dim=lap_evalues[dimension]
lap_infty=np.linalg.norm(lap, np.inf)
lap_fro=np.linalg.norm(lap, 'fro')

ridge_bound_array=np.zeros(item_num)
graph_ridge_bound_array=np.zeros(item_num)
for i in range(item_num):
	lam=lambda_(noise_level, d, user_num, dimension, i+dimension)
	lam=0.001
	x=item_f[:i+dimension, :]
	Sigma=np.cov(x.T)
	u,s,v=np.linalg.svd(Sigma)
	sigma_min=np.min(s)
	k=sigma_min*10
	ridge_bound_array[i]=ridge_bound_fro(lam, rank, I_user_fro, I_min, k)
	graph_ridge_bound_array[i]=graph_ridge_bound_fro(lam, rank, lap_user_fro, lap_min, k)

plt.figure()
# plt.plot(ridge_bound_array, label='ridge')
plt.plot(graph_ridge_bound_array, label='graph ridge')
plt.legend(loc=0, fontsize=12)
plt.show()




