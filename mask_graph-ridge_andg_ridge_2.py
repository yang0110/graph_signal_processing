import numpy as np 
import cvxpy as cp 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from utils import *
from sklearn import datasets
path='../results/Graph_ridge_results/'

np.random.seed(seed=2019)

user_num=50
item_num=200
dimension=10
noise_level=0.75
lam=0.01
I=np.identity(user_num)
user_f=np.random.normal(size=(user_num, dimension))
user_f,_=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
adj=ori_adj.copy()
thrs=np.round((min_adj+max_adj)/2, decimals=2)
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)

##generate item_f
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

## signal 
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 

## with mask 
mask=np.random.randint(2, size=(user_num, item_num))

ridge_error_array=np.zeros(item_num)
graph_ridge_error_array=np.zeros(item_num)
ridge_res=np.zeros((user_num, dimension))
graph_ridge_res=np.zeros((user_num, dimension))
for i in range(item_num):
	print('i', i)
	M=np.zeros((user_num,item_num))
	M[:,:i+dimension]=mask[:,:i+dimension]
	if (i%1==0) or (i==0):
		x=item_f.copy()
		y=noisy_signal.copy()
		ridge_res=ridge_mask_convex(user_num, dimension, I, x, y, lam, M)
		graph_ridge_res=graph_ridge_mask_convex(user_num, dimension, lap, x, y, lam, M)
		ridge_error_array[i]=np.linalg.norm(ridge_res-user_f, 'fro')
		graph_ridge_error_array[i]=np.linalg.norm(graph_ridge_res-user_f, 'fro')
	else:
		ridge_error_array[i]=ridge_error_array[i-1]
		graph_ridge_error_array[i]=graph_ridge_error_array[i-1]



plt.figure()
plt.plot(ridge_error_array[dimension:], 'r+-', markevery=0.1, label='ridge')
plt.plot(graph_ridge_error_array[dimension:],'b', label='graph ridge')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Sample size', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.title('user num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
plt.savefig(path+'mask_graph_ridge_vs_ridge_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()



