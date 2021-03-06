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


user_num=50
dimension=10
item_num=2000
noise_level=0.1
d=2

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
Sigma=np.cov(item_f.T)
u,s,v=np.linalg.svd(Sigma)
sigma_min=np.min(s)
k=sigma_min/18

user_f=np.random.normal(size=(user_num, dimension))
user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
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
	lam2=lam
	lam_list[i]=lam
	ridge_array[i]=ridge_bound_fro(lam, rank, I_user_fro, I_2, k)
	graph_ridge_array[i]=graph_ridge_bound_fro(lam2, rank, lap_user_fro, lap_min, k)
	graph_ridge_simple_array[i]=graph_ridge_bound_fro(lam2, rank, lam_user_fro, lap_min, k)

plt.figure()
plt.plot(ridge_array, label='ridge')
plt.plot(graph_ridge_array, label='graph ridge')
plt.plot(graph_ridge_simple_array,label='graph ridge simple')
plt.xlabel('sample size', fontsize=12)
plt.ylabel('theoretical bound', fontsize=12)
plt.title('same lambda', fontsize=12)
plt.legend(loc=0,fontsize=12)
plt.savefig(path+'lap_1_lap_lam_same_ridge_lam_theoretical_bound_ridge_gr_grs'+'.png', dpi=200)
plt.show()

plt.figure()
plt.plot(ridge_array, label='ridge')
plt.legend(loc=0,fontsize=12)
plt.show()

plt.figure()
plt.plot(lam_list, label='lam')
plt.legend(loc=0, fontsize=12)
plt.show()


plt.figure()
plt.plot(lam_list*lap_2, label='lam*lap_min')
plt.legend(loc=0, fontsize=12)
plt.show()

plt.figure()
plt.plot(lap_evalues, label='lap_evalues')
plt.legend(loc=0, fontsize=12)
plt.show()

cluster_std_list=np.arange(0.001, 10, 0.1)
ori_user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=1, shuffle=False, random_state=2019)
fro_list=np.zeros(len(cluster_std_list))
lam_list=np.zeros(len(cluster_std_list))
for i, cluster_std in enumerate(cluster_std_list):
	user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=cluster_std, shuffle=False, random_state=2019)
	user_f=Normalizer().fit_transform(user_f)
	adj=rbf_kernel(user_f)
	lap=csgraph.laplacian(adj, normed=False)
	lap_evalues, lap_evec=np.linalg.eig(lap)
	Lambda=np.diag(lap_evalues)
	lap_user_fro=np.linalg.norm(np.dot(lap, ori_user_f))
	lam_user_fro=np.linalg.norm(np.dot(Lambda, ori_user_f))
	fro_list[i]=lap_user_fro
	lam_list[i]=lam_user_fro

# plt.plot(cluster_std_list, lam_list, label='Lambda')
plt.plot(cluster_std_list, fro_list, label='Lap')
plt.legend(loc=0, fontsize=12)
plt.title('cluster_std=1', fontsize=12)
plt.xlabel('cluster_std', fontsize=12)
plt.ylabel('||L theta||_F', fontsize=12)
plt.savefig(path+'dot_product_of_lap_and_user_f'+'.png', dpi=200)
plt.show()





