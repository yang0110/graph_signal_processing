import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/code/')
import datetime 
import networkx as nx
from utils import *
path='../results/Graph_ridge_results/'

np.random.seed(seed=2019)

user_num=200
item_num=200
dimension=10
noise_level=0.25
g_lambda=0.01

node_f=np.random.normal(size=(user_num, dimension))
node_f=Normalizer().fit_transform(node_f)
ori_adj=rbf_kernel(node_f)
lap=csgraph.laplacian(ori_adj, normed=False)
cov=np.linalg.pinv(lap)
user_f=np.random.multivariate_normal(np.zeros(user_num), cov, size=dimension).T 
user_f=Normalizer().fit_transform(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
thrs_list=np.round(np.linspace(min_adj, max_adj, 4), decimals=2)

# user_f=np.random.normal(size=(user_num, dimension))
# user_f=Normalizer().fit_transform(user_f)
# ori_adj=rbf_kernel(user_f)
# min_adj=np.min(ori_adj)
# max_adj=np.max(ori_adj)
# thrs_list=np.round(np.linspace(min_adj, max_adj, 4), decimals=2)

##generate item_f
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

## signal 
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 

## slyvetser equation
re=np.zeros(item_num)
for i in range(item_num):
	print('Ridge i', i)
	x=item_f[:i+1, :]
	y=noisy_signal[:, :i+1]
	ridge=np.dot(np.linalg.inv(np.dot(x.T, x)+g_lambda*np.identity(dimension)), np.dot(x.T, y.T)).T
	re[i]=np.linalg.norm(ridge-user_f, 'fro')

ge=np.zeros((len(thrs_list), item_num))
edge_num_list=[]
for index, thrs in enumerate(thrs_list):
	print('thrs', thrs)
	adj=ori_adj.copy()
	adj[adj<=thrs]=0
	graph, edge_num=create_networkx_graph(user_num, adj)
	edge_num_list.extend([edge_num])
	lap=csgraph.laplacian(adj, normed=False)+np.identity(user_num)
	for i in range(item_num):
		print('Graph ridge', thrs, i)
		x=item_f[:i+1, :]
		y=noisy_signal[:, :i+1]
		A=g_lambda*lap 
		B=np.dot(x.T, x)
		C=np.dot(y, x)
		g_ridge=scipy.linalg.solve_sylvester(A, B, C)
		ge[index, i]=np.linalg.norm(g_ridge-user_f, 'fro')

plt.figure(figsize=(5,5))
plt.plot(re, 'k+', label='Ridge')
for i in range(len(thrs_list)):
	plt.plot(ge[i], label='Graph-ridge, T=%s, E=%s'%(thrs_list[i], edge_num_list[i]))

plt.xlabel('Training set size', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.title('Use num=%s, Noise=%s'%(user_num, noise_level), fontsize=12)
plt.legend(loc=0, fontsize=10)
plt.savefig(path+'random_model/'+'learning_error_user_num_noise_%s_%s'%(user_num, noise_level)+'.png', dpi=100)
plt.show()
