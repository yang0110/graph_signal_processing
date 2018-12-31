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
path='../results/LapUCB_results/'

user_num=30
item_num=300
dimension=10
iteration=1000
noise_level=0.1
alpha=0.01

confidence_inter=0.05

item_f=datasets.make_low_rank_matrix(n_samples=item_num,n_features=dimension, effective_rank=10, random_state=2019)
item_f=Normalizer().fit_transform(item_f)

# node_f=np.random.normal(size=(user_num, dimension))
# node_f=Normalizer().fit_transform(node_f)
# ori_adj=rbf_kernel(node_f)
# lap=csgraph.laplacian(ori_adj, normed=False)
# cov=np.linalg.pinv(lap)
# user_f=np.random.multivariate_normal(np.zeros(user_num), cov, size=dimension).T
# user_f=Normalizer().fit_transform(user_f)

user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)
ori_adj=rbf_kernel(user_f)
min_adj=np.round(np.min(ori_adj), decimals=2)
max_adj=np.round(np.max(ori_adj), decimals=2)
thrs_list=np.linspace(min_adj, 0.99, 3)

clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise

pool_size=25
user_array=np.random.choice(np.arange(user_num), size=iteration)
item_array=np.random.choice(np.arange(item_num), size=iteration*pool_size).reshape((iteration, pool_size))

linucb=LinUCB(user_num, item_num, dimension, pool_size, user_f, item_f,noisy_signal, alpha, confidence_inter)
l_cum_regret, l_error_list, l_e_array=linucb.run(user_array, item_array, iteration)

cum_regret_array={}
error_array={}
edge_num_list=[]
for thrs in thrs_list:
	print('thrs', thrs)
	adj=ori_adj.copy()
	adj[adj<=thrs]=0
	graph, edge_num=create_networkx_graph(user_num, adj)
	edge_num_list.extend([edge_num])
	lap=csgraph.laplacian(adj, normed=False)+np.identity(user_num)
	graph_ridge=Graph_ridge(user_num, item_num, dimension, lap, adj, pool_size, user_f, item_f, noisy_signal, alpha, confidence_inter)
	g_cum_regret, g_error_list, g_e_array, trace_list=graph_ridge.run(user_array, item_array, iteration)
	cum_regret_array[thrs]=g_cum_regret
	error_array[thrs]=g_error_list


plt.figure(figsize=(5,5))
plt.plot(l_cum_regret, 'r', label='LinUCB')
for thrs, edge_num in zip(thrs_list, edge_num_list):
	plt.plot(cum_regret_array[thrs], label='LapUCB thres=%s, edge num=%s'%(thrs, edge_num))

plt.legend(loc=0, fontsize=10)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cum Regret', fontsize=12)
plt.savefig(path+'consistent_model/'+'cum_regret_user_num_noise_%s_%s'%(user_num, noise_level)+',png', dpi=100)
plt.show()


plt.figure(figsize=(5,5))
plt.plot(l_error_list, 'r', label='LinUCB')
for thrs, edge_num in zip(thrs_list, edge_num_list):
	plt.plot(error_array[thrs], label='LapUCB thres=%s, edge num=%s'%(thrs, edge_num))

plt.legend(loc=0, fontsize=10)
plt.ylabel('Learning Error (Frobenius Norm)')
plt.xlabel('Iteration', fontsize=12)
plt.savefig(path+'consistent_model/'+'cum_regret_user_num_noise_%s_%s'%(user_num, noise_level)+',png', dpi=100)
plt.show()



