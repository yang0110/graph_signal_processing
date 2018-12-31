import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge
from sklearn import datasets
path='../results/LapUCB_results/'

user_num=50
item_num=300
dimension=10
iteration=5000
noise_level=0.01
alpha=0.01

confidence_inter=0.05

item_f=datasets.make_low_rank_matrix(n_samples=item_num,n_features=dimension, effective_rank=10, random_state=2019)
item_f=MinMaxScaler().fit_transform(item_f)

# node_f=np.random.normal(size=(user_num, dimension))
# node_f=Normalizer().fit_transform(node_f)
# adj=rbf_kernel(node_f)
# thrs=0.8
# adj[adj<=thrs]=0.0
# lap=csgraph.laplacian(adj, normed=False)+np.identity(user_num)
# cov=np.linalg.pinv(lap)
# user_f=np.random.multivariate_normal(np.zeros(user_num), cov, size=dimension).T
# user_f=Normalizer().fit_transform(user_f)

user_f=np.random.normal(size=(user_num, dimension))
user_f=MinMaxScaler().fit_transform(user_f)
adj=rbf_kernel(user_f)
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise

pool_size=25
user_array=np.random.choice(np.arange(user_num), size=iteration)
item_array=np.random.choice(np.arange(item_num), size=iteration*pool_size).reshape((iteration, pool_size))

linucb=LinUCB(user_num, item_num, dimension, pool_size, user_f, item_f,noisy_signal, alpha, confidence_inter)
l_cum_regret, l_error_list, l_e_array=linucb.run(user_array, item_array, iteration)


graph_ridge=Graph_ridge(user_num, item_num, dimension, lap, adj, pool_size, user_f, item_f, noisy_signal, alpha, confidence_inter)
g_cum_regret, g_error_list, g_e_array, trace_list=graph_ridge.run(user_array, item_array, iteration)


plt.figure(figsize=(5,5))
plt.plot(l_cum_regret, 'r', label='LinUCB')
plt.plot(g_cum_regret, 'y', label='LapUCB')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cum regret', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.savefig(path+'random_model/'+'cum_regret_user_num_noise_%s_%s'%(user_num, noise_level)+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(l_error_list, 'r', label='LinUCB')
plt.plot(g_error_list, 'y', label='LapUCB')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.savefig(path+'random_model/'+'learing_error_user_num_noise_%s_%s'%(user_num, noise_level)+'.png', dpi=100)
plt.show()


plt.plot(trace_list)
plt.show()

labels = ['user-1', 'user-2', 'user-3', 'user-4', 'user-5']
plt.figure(figsize=(5,5))
for y_arr, label in zip(l_e_array.T, labels):
    plt.plot(y_arr, label=label)

plt.legend(loc=0, fontsize=12)
plt.xlabel('iteration', fontsize=12)
plt.ylabel('Cum regret', fontsize=12)
plt.title('LinUCB \n Noise= %s, threshold= %s'%(noise_level, thrs), fontsize=12)
plt.savefig(path+'LinUCB-top-5_users_noise_%s_threshold_%s'%(noise_level, thrs)+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
for g_arr, label in zip(g_e_array.T, labels):
    plt.plot(g_arr, label=label)

plt.legend(loc=0, fontsize=12)
plt.xlabel('iteration', fontsize=12)
plt.ylabel('Cum regret', fontsize=12)
plt.title('LapUCB \n Noise= %s, threshold= %s'%(noise_level, thrs), fontsize=12)
plt.savefig(path+'LapUCB-top-5_users_noise_%s_threshold_%s'%(noise_level, thrs)+'.png', dpi=100)
plt.show()
