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
path='../results/Graph_ridge_results/error_bound/'

np.random.seed(2019)

def ridge_bound(noise_level, rank, lap_fro, user_fro,lap_min, k, d, user_num, dimension, item_num):
	lam=8*np.sqrt(noise_level)*np.sqrt(d)*np.sqrt(user_num+dimension)/(user_num*item_num)
	bound=lam*(np.sqrt(rank)+2*lap_fro*user_fro)/(k+lam*lap_min)
	return bound

def graph_ridge_bound(noise_level, rank,lap_fro, user_fro, lap_min, k, d, user_num, dimension, item_num):
	lam=8*np.sqrt(noise_level)*np.sqrt(d)*np.sqrt(user_num+dimension)/(user_num*item_num)
	bound=lam*(np.sqrt(rank)+2*lap_fro*user_fro)/(k+lam*lap_min)
	return bound 

user_num=200
dimension=25
item_num=300
noise_level=0.1
d=3

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

user_f=np.random.normal(size=(user_num, dimension))
#user_f,_=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=10)
user_f=Normalizer().fit_transform(user_f)
rank=np.linalg.matrix_rank(user_f)

user_fro=np.linalg.norm(user_f, 'fro')
ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
thrs_list=np.round(np.linspace((min_adj+max_adj)/2, max_adj, 5), decimals=4)

I=np.identity(user_num)
I_ev, I_evc=np.linalg.eig(I)
I_min=np.max(I_ev)
I_fro=np.linalg.norm(I, 'fro')

ridge_bound_array=np.zeros(item_num)
graph_ridge_bound_array=np.zeros((len(thrs_list), item_num))
edge_num_list=np.zeros(len(thrs_list))
for ind, thrs in enumerate(thrs_list):
	print('thrs', thrs)
	adj=ori_adj.copy()
	np.fill_diagonal(adj, 0)
	adj[adj<=thrs]=0
	graph, edge_num=create_networkx_graph(user_num, adj)
	edge_num_list[ind]=edge_num
	lap=csgraph.laplacian(adj, normed=False)+np.identity(user_num)
	eigenvalues, eigenvectors=np.linalg.eig(lap)
	lap_min=np.max(eigenvalues)
	lap_fro=np.linalg.norm(lap, 'fro')
	for i in range(item_num):
		x=item_f[:i+dimension,:]
		Sigma=np.cov(x.T)
		u, s, v=np.linalg.svd(Sigma)
		sigma_min=np.min(s)
		k=sigma_min/18
		print('thrs, i', thrs, i)
		ridge_bound_array[i]=ridge_bound(noise_level, rank, I_fro, user_fro, I_min, k, d, user_num, dimension, i+dimension)
		graph_ridge_bound_array[ind, i]=graph_ridge_bound(noise_level, rank, lap_fro, user_fro, lap_min, k, d, user_num, dimension, i+dimension)

plt.figure()
plt.plot(ridge_bound_array, 'k+-',markevery=0.1, label='ridge bound')
for index, thrs in enumerate(thrs_list):
	plt.plot(graph_ridge_bound_array[index,], label='T=%s, E=%s'%(thrs, edge_num_list[index]))

plt.legend(loc=0, fontsize=12)
plt.xlabel('Sample size', fontsize=12)
plt.ylabel('Bound', fontsize=12)
plt.title('Theoretical Bound', fontsize=12)
plt.show()


# adj=rbf_kernel(user_f)
# min_adj=np.min(adj)
# max_adj=np.max(adj)
# thrs_list=np.round(np.linspace(min_adj,1.0, 10), decimals=4)

# graph_ridge_bound_array={}
# edge_num_list=[]
# for thrs in thrs_list:
# 	print('thrs', thrs)
# 	new_adj=adj.copy()
# 	new_adj[new_adj<thrs]=0
# 	np.fill_diagonal(new_adj, 0)
# 	graph, edge_num=create_networkx_graph(user_num, new_adj)
# 	edge_num_list.extend([edge_num])
# 	I=np.identity(user_num)
# 	I_fro=np.linalg.norm(I, 'fro')
# 	lap=csgraph.laplacian(new_adj, normed=False)
# 	eigenvalues, eigvectors=np.linalg.eig(lap)
# 	lap_fro=np.linalg.norm(lap,'fro')
# 	lap_min=np.min(eigenvalues)
# 	ridge_bound_list=[]
# 	graph_ridge_bound_list=[]
# 	for i in range(item_num):
# 		ridge_b=ridge_bound(noise_level, rank, I_fro, user_fro, k, d, user_num, dimension, i+1)
# 		ridge_bound_list.extend([ridge_b])
# 		graph_ridge_b=graph_ridge_bound(noise_level, rank, user_fro, k, lap_fro, lap_min, d, user_num, dimension, i+1)
# 		graph_ridge_bound_list.extend([graph_ridge_b])
# 	graph_ridge_bound_array[thrs]=graph_ridge_bound_list


# plt.figure()
# plt.plot(ridge_bound_list[dimension:],'r+', markevery=0.01, label='ridge')
# for thrs, edge in zip(thrs_list[5:], edge_num_list[5:]):
# 	plt.plot(graph_ridge_bound_array[thrs][dimension:], label='T='+str(thrs)+','+'E='+str(edge))

# plt.legend(loc=0, fontsize=12)
# plt.title('user num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
# plt.xlabel('item num', fontsize=12)
# plt.ylabel('bound', fontsize=12)
# plt.savefig(path+'Theoretical_bound_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=100)
# plt.show()


# user_f=np.random.normal(size=(user_num, dimension))
# user_f=Normalizer().fit_transform(user_f)
# adj=rbf_kernel(user_f)
# min_adj=np.min(adj)
# max_adj=np.max(adj)
# thrs_list=np.round(np.linspace(min_adj, max_adj, 100), decimals=5)
# ratio_list=[]
# for thrs in thrs_list:
# 	new_adj=adj.copy()
# 	new_adj[new_adj<=thrs]=0
# 	lap=csgraph.laplacian(new_adj, normed=False)
# 	lap_infty=np.linalg.norm(lap, np.inf)
# 	lap_trace=np.trace(lap)
# 	ratio=lap_infty/lap_trace
# 	ratio_list.extend([ratio])

# plt.figure()
# plt.plot(thrs_list, ratio_list)
# plt.show()
