import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge, Graph_ridge_simple
from sklearn import datasets
from utils import *
path='../results/LapUCB_results/'

user_num=20
item_num=100
dimension=10
iteration=5000
noise_level=0.1

confidence_inter=0.05

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

user_f=np.random.normal(size=(user_num, dimension))
user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
adj=rbf_kernel(user_f)
min_adj=np.min(adj)
max_adj=np.max(adj)
thrs=np.round((min_adj+max_adj)/2, decimals=2)
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)
lap_evalues, lap_evectors=np.linalg.eig(lap)
lap_evalues=np.sort(lap_evalues)
evalues_matrix=np.diag(lap_evalues)

clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise

pool_size=10
user_array=np.random.choice(np.arange(user_num), size=iteration)
item_array=np.random.choice(np.arange(item_num), size=iteration*pool_size).reshape((iteration, pool_size))

lin_lam=0.01
graph_lam=0.01

linucb=LinUCB(user_num, item_num, dimension, pool_size, user_f, item_f,noisy_signal, lin_lam, confidence_inter)
l_cum_regret, l_error_list, l_e_array=linucb.run(user_array, item_array, iteration)

graph_ridge=Graph_ridge(user_num, item_num, dimension, lap, adj, pool_size, user_f, item_f, noisy_signal, graph_lam, confidence_inter)
g_cum_regret, g_error_list, g_e_array=graph_ridge.run(user_array, item_array, iteration)

graph_ridge_s=Graph_ridge_simple(user_num, item_num, dimension, evalues_matrix, adj, pool_size, user_f, item_f, noisy_signal, graph_lam, confidence_inter)
g_cum_regret2, g_error_list2, g_e_array2=graph_ridge_s.run(user_array, item_array, iteration)


plt.figure(figsize=(5,5))
plt.plot(l_cum_regret, 'r', label='LinUCB')
plt.plot(g_cum_regret, 'y', label='LapUCB')
plt.plot(g_cum_regret2, 'g', label='LamUCB')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cum regret', fontsize=12)
plt.title('User num=%s, noise=%s, T=%s'%(user_num, noise_level, thrs), fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.savefig(path+'random_model/'+'LinUCB_vs_LapUCB_cum_regret_user_num_noise_thrs_%s_%s_%s'%(user_num, noise_level, thrs)+'.png', dpi=300)
plt.show()

plt.figure(figsize=(5,5))
plt.plot(l_error_list, 'r', label='LinUCB')
plt.plot(g_error_list, 'y', label='LapUCB')
plt.plot(g_error_list2, 'g',label='LamUCB')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.title('User num=%s, noise=%s, T=%s'%(user_num, noise_level, thrs), fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.savefig(path+'random_model/'+'LinUCB_vs_LapUCB_learning_error_user_num_noise_thrs_%s_%s_%s'%(user_num, noise_level, thrs)+'.png', dpi=300)
plt.show()


user_frequency=np.zeros(user_num)
for i in range(user_num):
	user_frequency[i]=np.sum(user_array==i)

sort_index=np.argsort(user_frequency)
labels=list(sort_index[:3])+list(sort_index[-3:])
small_user_fre=user_frequency[labels]

label_dict={}
for i in range(len(labels)):
	label_dict[i]=str(i)

color_list=['r']*3+['y']*3
small_adj=adj[labels][:, labels]
np.fill_diagonal(small_adj, 0)
pos=np.random.uniform(size=(len(labels), 2))
graph, edge_num=create_networkx_graph(len(labels), small_adj)
edge_weight=small_adj[np.triu_indices(len(labels),1)]
edge_color=edge_weight[edge_weight>=0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=800, node_color=color_list)
edges=nx.draw_networkx_edges(graph, pos, width=2, alpha=1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
nx.draw_networkx_labels(graph, pos, label_dict, fontsize=12)
plt.axis('off')
plt.savefig(path+'random_model/'+'top_user_graph_and_edge_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()

ind=np.arange(6)
plt.bar(ind, small_user_fre, align='center')
plt.xticks(ind, labels)
plt.xlabel('User Index', fontsize=12)
plt.ylabel('User Frequency', fontsize=12)
plt.savefig(path+'random_model/'+'user_frequency_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()


fig, (ax1, ax2)=plt.subplots(1,2, figsize=(8,5))
ax1.plot(l_e_array[:,:len(labels)+1])
ax1.legend(loc=0, labels=labels, fontsize=10)
ax1.set_ylim([0.0, 1.2])
ax1.set_title('LinUCB',fontsize=12)
ax2.plot(g_e_array[:, :len(labels)+1])
ax2.set_ylim([0.0, 1.2])
ax2.legend(loc=0, labels=labels, fontsize=10)
ax2.set_title('LapUCB', fontsize=12)
plt.xlabel('User num=%s, noise=%s'%(user_num, noise_level), fontsize=10)
plt.savefig(path+'random_model/'+'LapUCB_VS_LinUCB_top_5_users_user_num_%s_noise_%s_thres_%s'%(user_num, noise_level, thrs)+'.png', dpi=300)
plt.show()
