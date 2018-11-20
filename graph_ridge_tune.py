import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import os 
os.chdir('../code/')
import datetime 
import networkx as nx
from utils import *


timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 
newpath='../results/'



user_num=10
item_num=100
dimension=10
noise_scale=0.1
p=0.8
user_f,item_f,pos,noisy_signal,adj,lap=generate_random_graph(user_num, item_num, 
	dimension, noise_scale)
# user_f,item_f,pos,noisy_signal,adj,lap=generate_GMRF(user_num, item_num, dimension, noise_scale)

g_lam_list=np.arange(1.6, 3, 0.2)
iteration=100

all_user=list(range(user_num))
all_item=list(range(item_num))
user_list=[]
item_list=[]

beta_graph_ridge=np.zeros((user_num, dimension))
user_dict={a:[] for a in all_user}
error_list_graph_ridge={lam:[] for lam in g_lam_list}

for i in range(iteration):
	print('iteration i=', i )
	if i<user_num:
		user=all_user[i]
		item=np.random.choice(all_item)
		user_dict[user].extend([item])
		item_sub_list=user_dict[user]
		user_list.extend([user])
		item_list.extend([item])
		user_list=list(np.unique(user_list))
		item_list=list(np.unique(item_list))
		user_nb=len(user_list)
	else:
		user=np.random.choice(all_user)
		item=np.random.choice(all_item)
		user_dict[user].extend([item])
		item_sub_list=user_dict[user]
		X=item_f[:, item_sub_list].T
		y=noisy_signal[user, item_sub_list]
		### Graph Ridge 
		user_list.extend([user])
		item_list.extend([item])
		user_list=list(np.unique(user_list))
		item_list=list(np.unique(item_list))
		user_nb=len(user_list)
		print('served user number', user_nb)
		item_nb=len(item_list)
		item_feature=item_f[:, item_list]
		signal=noisy_signal[user_list][:,item_list]
		mask=np.ones((user_nb, item_nb))
		for ind, j in enumerate(user_list):
			remove_list=[x for x, xx in enumerate(item_list) if xx not in user_dict[j]]
			mask[ind, remove_list]=0
		signal=signal*mask
		sub_lap=lap[user_list][:,user_list]
		print('sub_lap.size', sub_lap.shape)
		for g_lambda in g_lam_list:
			u_f=graph_ridge(user_nb, item_nb, dimension, sub_lap, item_feature, mask, signal, g_lambda)
			beta_graph_ridge[user_list]=u_f.copy()
			error_graph_ridge=np.linalg.norm(beta_graph_ridge-user_f)
			error_list_graph_ridge[g_lambda].extend([error_graph_ridge])

plt.figure()
for l in g_lam_list:
	plt.plot(error_list_graph_ridge[l][user_num:], label='lambda=%s'%(l))
plt.legend(loc=0)
plt.show()




graph, edge_num=create_networkx_graph(user_num, adj)
edge_weight=adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=50, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.savefig(newpath+str(timeRun)+'_'+'graph_user_num_%s_egde_num_%s'%(user_num, edge_num)+'.png', dpi=100)
plt.clf()


for dim in range(dimension):
	plt.figure(figsize=(5,5))
	nodes=nx.draw_networkx_nodes(graph, pos, node_size=50, node_color=user_f[:,dim], cmap=plt.cm.jet)
	edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
	plt.axis('off')
	plt.savefig(newpath+str(timeRun)+'user_feature_smoothness_dimension_%s'%(dim)+'.png', dpi=100)
	plt.clf()





