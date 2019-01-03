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

user_num=100
item_num=300
dimension=25
noise_level=2
d=3

I=np.identity(user_num)
user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)
ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
thrs_list=np.round(np.linspace((min_adj+max_adj)/2, max_adj, 6), decimals=4)

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
	g_lambda=8*np.sqrt(noise_level)*np.sqrt(d)*np.sqrt(user_num+dimension)/(user_num*(i+dimension))
	g_lambda=0.1
	print('Ridge i', i)
	x=item_f[:i+dimension, :]
	y=noisy_signal[:, :i+dimension]
	#ridge=np.dot(np.linalg.inv(np.dot(x.T, x)+g_lambda*np.identity(dimension)), np.dot(x.T, y.T)).T
	A=g_lambda*I
	B=np.dot(x.T, x)
	C=np.dot(y,x)
	ridge=scipy.linalg.solve_sylvester(A, B, C)
	re[i]=np.linalg.norm(ridge-user_f, 'fro')

ge=np.zeros((len(thrs_list), item_num))
edge_num_list=[]
for index, thrs in enumerate(thrs_list):
	print('thrs', thrs)
	adj=ori_adj.copy()
	np.fill_diagonal(adj,0)
	adj[adj<=thrs]=0
	graph, edge_num=create_networkx_graph(user_num, adj)
	edge_num_list.extend([edge_num])
	lap=csgraph.laplacian(adj, normed=False)
	for i in range(item_num):
		g_lambda=8*np.sqrt(noise_level)*np.sqrt(d)*np.sqrt(user_num+dimension)/(user_num*(i+dimension))
		g_lambda=0.1
		print('Graph ridge', thrs, i)
		x=item_f[:i+dimension, :]
		y=noisy_signal[:, :i+dimension]
		A=g_lambda*lap 
		B=np.dot(x.T, x)
		C=np.dot(y, x)
		g_ridge=scipy.linalg.solve_sylvester(A, B, C)
		ge[index, i]=np.linalg.norm(g_ridge-user_f, 'fro')

plt.figure(figsize=(5,5))
plt.plot(re[dimension:], 'k+', markevery=0.1, label='Ridge')
for i in range(len(thrs_list)):
	plt.plot(ge[i][dimension:], label='Graph-ridge, T=%s, E=%s'%(thrs_list[i], edge_num_list[i]))

plt.xlabel('Training set size', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.title('Use num=%s, Noise=%s'%(user_num, noise_level), fontsize=12)
plt.legend(loc=0, fontsize=10)
plt.savefig(path+'random_model/'+'learning_error_user_num_noise_%s_%s'%(user_num, noise_level)+'.png', dpi=100)
plt.show()
