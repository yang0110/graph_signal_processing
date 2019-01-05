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

user_num=10
item_num=100
iteration=2000
dimension=5
noise_level=0.5

I=np.identity(user_num)
user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)
adj=rbf_kernel(user_f)
min_adj=np.min(adj)
max_adj=np.max(adj)
thrs=np.round((min_adj+max_adj)/2, decimals=2)
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 

random_user_array=np.random.choice(range(user_num), size=iteration)
random_item_array=np.random.choice(range(item_num), size=iteration)

lam_list=list(np.round(np.linspace(0.01, 0.1, 5), decimals=3))+list(np.round(np.linspace(0.1, 1, 5), decimals=2))

ridge_error_array=np.zeros((len(lam_list), iteration))
graph_ridge_error_array=np.zeros((len(lam_list), iteration))
mask=np.zeros((user_num, item_num))
ridge_res=np.zeros((user_num, dimension))
graph_ridge_res=np.zeros((user_num, dimension))
for i in range(iteration):
	print('i', i)
	user=random_user_array[i]
	item=random_item_array[i]
	served_items=list(np.unique(random_item_array[:i+1]))
	mask[user, item]=1
	neighbors=list(np.where(adj[user]>0)[0])
	print('neighbors num', len(neighbors))
	L=lap[neighbors][:, neighbors]
	M=mask[neighbors][:, served_items]
	I=np.identity(len(neighbors))
	for index, lam in enumerate(lam_list):
		if (i%10==0) or (i==0):
			x=item_f[served_items]
			y=noisy_signal[neighbors][:, served_items]
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
plt.plot(ridge_error_array[min_ridge][20*dimension:], 'r+-', markevery=0.1,label='ridge, lambda=%s'%(ridge_lam))
plt.plot(graph_ridge_error_array[min_graph][20*dimension:],'b', label='graph ridge, lambda=%s'%(graph_lam))
plt.legend(loc=0, fontsize=12)
plt.xlabel('Sample size', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.title('user num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
plt.savefig(path+'mask_graph_ridge_vs_ridge_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()


