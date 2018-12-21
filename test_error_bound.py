import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import os 
#os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from utils import *

#path='Documents/research/results/'

def error_bound(min_e, g_lambda, trace_l):
	k=min_e/18
	bound=g_lambda/(2*(k+g_lambda*trace_l))
	return bound 

def error_bound_2(lam, trace_l, min_e):
	bound=lam/(2*((min_e/18)+lam*trace_l))
	return bound 

def Lambda(max_e, noise_level, user_num, item_num, dimension):
	lam=10*noise_level*np.sqrt(max_e)*np.sqrt((user_num+dimension)/item_num)
	return lam 

user_num=10
dimension=5
item_num=100
noise_level=0.1
iteration=item_num
g_lambda=0.1

user_f=np.random.normal(size=(user_num, dimension))
item_f=np.random.normal(size=(item_num, dimension))
user_f=Normalizer().fit_transform(user_f)
item_f=Normalizer().fit_transform(item_f)

ori_adj=rbf_kernel(user_f)
thrs_list=np.arange(np.min(ori_adj)+0.1, np.max(ori_adj)-0.1, 0.1)
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 
error_list={}
for thrs in thrs_list:
	print('thrs', thrs)
	adj=ori_adj.copy()
	adj[adj<thrs]=0
	lap=csgraph.laplacian(adj, normed=False)
	trace_l=np.trace(lap)
	es_userf=np.zeros((user_num, dimension))
	error_list[thrs]=[]
	for i in range(iteration):
		print('iter', i)
		x=item_f[:i+1, :]
		y=noisy_signal[:,:i+1]
		Sigma=np.dot(x.T, x)
		e_s=np.linalg.eigvals(Sigma)
		max_e=np.max(e_s)
		min_e=np.min(e_s)
		g_lambda=Lambda(max_e, noise_level, user_num, item_num, dimension)
		g_lambda=0.1
		es_userf=graph_ridge_no_mask(user_num, dimension, lap, x, y, g_lambda)
		error=np.linalg.norm(es_userf-user_f, 'fro')
		error_list[thrs].extend([error])

plt.figure(figsize=(5,5))
for thrs in thrs_list:
	plt.plot(error_list[thrs], label='thrs=%s'%(thrs))	
plt.legend(loc=0)
plt.show()


# plt.figure(figsize=(5,5))
# plt.plot(error_list, label='Error')
# plt.plot(bound_list, label='Bound')
# plt.legend(loc=0, fontsize=12)
# plt.xlabel('Iteration= # of training samples', fontsize=12)
# plt.ylabel('Error', fontsize=12)
# plt.show()

# plt.figure(figsize=(5,5))
# plt.plot(bound_list, label='Bound')
# plt.xlabel('Iteration= # of training samples', fontsize=12)
# plt.ylabel('Error Bound', fontsize=12)
# plt.show()


pos=user_f.copy()
np.fill_diagonal(adj, 0)
graph, edge_num=create_networkx_graph(user_num, adj)
edge_weight=adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
node_color=clear_signal[:,0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=50, node_color=node_color, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.show()









