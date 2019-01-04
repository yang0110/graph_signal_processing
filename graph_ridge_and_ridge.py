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

user_num=100
item_num=500
dimension=25
noise_level=1

I=np.identity(user_num)
user_f=np.random.normal(size=(user_num, dimension))
#user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
adj=ori_adj.copy()
thrs=(min_adj+max_adj)/2
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)


# plt.figure(figsize=(5,5))
# plt.pcolor(adj)
# plt.colorbar()
# plt.title('Adj matrix', fontsize=12)
# plt.savefig(path+'adj_matrix_random_user'+'.png', dpi=300)
# plt.show()
##generate item_f
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

## signal 
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 

lam_list=list(np.round(np.linspace(0.001, 0.01, 5), decimals=3))+list(np.round(np.linspace(0.01, 0.1, 5), decimals=3))+list(np.round(np.linspace(0.1, 1, 5), decimals=2))

# ridge_error_array=np.zeros((len(lam_list), item_num))
# graph_ridge_error_array=np.zeros((len(lam_list), item_num))
# for i in range(item_num):
# 	for index, lam in enumerate(lam_list):
# 		print('i, lam', i, lam)
# 		x=item_f[:i+dimension, :]
# 		y=noisy_signal[:, :i+dimension]
# 		A=lam*lap
# 		B=np.dot(x.T, x)
# 		C=np.dot(y,x)
# 		graph_ridge=scipy.linalg.solve_sylvester(A, B, C)
# 		graph_ridge_error_array[index, i]=np.linalg.norm(graph_ridge-user_f, 'fro')
# 		AA=lam*I 
# 		ridge=scipy.linalg.solve_sylvester(AA, B, C)
# 		ridge_error_array[index, i]=np.linalg.norm(ridge-user_f, 'fro')


# labels=lam_list
# plt.figure()
# plt.plot(ridge_error_array.T[dimension:])
# plt.title('ridge \n user_num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
# plt.legend(loc=0, labels=labels)
# plt.xlabel('Sample size', fontsize=12)
# plt.ylabel('Learning Error', fontsize=12)
# plt.savefig(path+'no_mask_ridge_tune_lam_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
# plt.show()


# plt.figure()
# plt.plot(graph_ridge_error_array.T[dimension:])
# plt.title('graph ridge \n user_num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
# plt.xlabel('Sample size', fontsize=12)
# plt.ylabel('Learning Error', fontsize=12)
# plt.legend(loc=0, labels=labels)
# plt.savefig(path+'no_mask_graph_ridge_tune_lam_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
# plt.show()


# min_ridge=np.argmin(ridge_error_array[:,-1])
# min_graph=np.argmin(graph_ridge_error_array[:,-1])
# ridge_lam=lam_list[min_ridge]
# graph_lam=lam_list[min_graph]

# plt.figure()
# plt.plot(ridge_error_array[min_ridge][dimension:], 'r+', markevery=0.1,label='ridge, lambda=%s'%(ridge_lam))
# plt.plot(graph_ridge_error_array[min_graph][dimension:],'b', label='graph ridge, lambda=%s'%(graph_lam))
# plt.legend(loc=0, fontsize=12)
# plt.xlabel('Sample size', fontsize=12)
# plt.ylabel('Learning Error', fontsize=12)
# plt.title('user num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
# plt.savefig(path+'no_mask_graph_ridge_vs_ridge_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
# plt.show()




## with mask 
iteration=np.int(item_num*user_num)
random_user_array=np.random.choice(range(user_num), size=iteration)
random_item_array=np.random.choice(range(item_num), size=iteration)
k=10 #number of neighbors to update

lam_list=list(np.round(np.linspace(0.01, 0.1, 5), decimals=3))+list(np.round(np.linspace(0.1, 2, 5), decimals=2))

ridge_error_array=np.zeros((len(lam_list), iteration))
graph_ridge_error_array=np.zeros((len(lam_list), iteration))
mask=np.zeros((user_num, item_num))
ridge_res=np.zeros((user_num, dimension))
graph_ridge_res=np.zeros((user_num, dimension))
for i in range(iteration):
	user=random_user_array[i]
	item=random_item_array[i]
	mask[user, item]=1
	neighbors=list(np.where(adj[user]>0)[0])[:k]
	L=lap[neighbors][:, neighbors]
	M=mask[neighbors]
	I=np.identity(len(neighbors))
	for index, lam in enumerate(lam_list):
		print('i, lam, n', i, lam, len(neighbors))
		if (i%50==0) or  (i==0):
			x=item_f.copy()
			y=noisy_signal[neighbors]
			ridge_res[neighbors]=ridge_mask_convex(len(neighbors), dimension, I, x, y, lam, M)
			graph_ridge_res[neighbors]=graph_ridge_mask_convex(len(neighbors), dimension, L, x, y, lam, M)
			ridge_error_array[index, i]=np.linalg.norm(ridge_res-user_f, 'fro')
			graph_ridge_error_array[index, i]=np.linalg.norm(graph_ridge_res-user_f, 'fro')
		else:
			ridge_error_array[index, i]=ridge_error_array[index, i-1]
			graph_ridge_error_array[index, i]=graph_ridge_error_array[index, i-1]


labels=lam_list
plt.figure()
plt.plot(ridge_error_array.T[dimension:])
plt.title('ridge \n user_num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
plt.legend(loc=0, labels=labels)
plt.xlabel('Sample size', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.savefig(path+'mask_ridge_tune_lam_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()


plt.figure()
plt.plot(graph_ridge_error_array.T[dimension:])
plt.title('graph ridge \n user_num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
plt.xlabel('Sample size', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.legend(loc=0, labels=labels)
plt.savefig(path+'mask_graph_ridge_tune_lam_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()


min_ridge=np.argmin(ridge_error_array[:,-1])
min_graph=np.argmin(graph_ridge_error_array[:,-1])
ridge_lam=lam_list[min_ridge]
graph_lam=lam_list[min_graph]

plt.figure()
plt.plot(ridge_error_array[min_ridge][dimension:], 'r+', markevery=0.1,label='ridge, lambda=%s'%(ridge_lam))
plt.plot(graph_ridge_error_array[min_graph][dimension:],'b', label='graph ridge, lambda=%s'%(graph_lam))
plt.legend(loc=0, fontsize=12)
plt.xlabel('Sample size', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.title('user num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
plt.savefig(path+'mask_graph_ridge_vs_ridge_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()


