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
user_num=50
item_num=100
dimension=10
# user_f,item_f,pos,ori_signal,adj,lap=generate_random_graph(user_num, item_num, dimension)
user_f,item_f,pos,ori_signal,adj,lap=generate_GMRF(user_num, item_num, dimension)
error_list={}
noise_list=[0.1, 0.25, 0.5]
for noise_scale in noise_list:
	print('noise_scale', noise_scale)
	error_list[noise_scale]=[]
	noisy_signal=ori_signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)

	g_lambda=2
	iteration=5000

	all_user=list(range(user_num))
	all_item=list(range(item_num))
	user_list=[]
	item_list=[]

	beta_gb=np.zeros((user_num, dimension))
	user_dict={a:[] for a in all_user}
	error_list_gb=[]

	for i in range(iteration):
		print('iteration i=', i )
		print('noise_scale=', noise_scale)
		if i<user_num:
			user=all_user[i]
			item=np.random.choice(all_item)
			user_dict[user].extend([item])
			item_sub_list=user_dict[user]
			user_list.extend([user])
			item_list.extend([item])
			user_list=list(np.unique(user_list))
			item_list=list(np.unique(item_list))
		else:
			user=np.random.choice(all_user)
			item=np.random.choice(all_item)
			user_dict[user].extend([item])
			item_sub_list=user_dict[user]
			user_list.extend([user])
			item_list.extend([item])
			user_list=list(np.unique(user_list))
			beta_gb=graph_ridge_iterative_model(item_f, noisy_signal, lap, beta_gb, dimension, user_list, item_list, user_dict, g_lambda)
			error_gb=np.linalg.norm(beta_gb-user_f)
			error_list[noise_scale].extend([error_gb])


plt.figure()
for n in noise_list:
	plt.plot(error_list[n], label='noise scale=%s'%(n))
plt.legend(loc=0)
plt.ylabel('MSE (Error)', fontsize=12)
plt.xlabel('#of sample (Size of training set)', fontsize=12)
plt.savefig(newpath+str(timeRun)+'Graph_ridge_iterative_error_tune_noise_user_num_%s'%(user_num)+'.png', dpi=100)
plt.show()





# plt.figure(figsize=(8,5))
# plt.plot(error_list_gb, label='Graph-ridge Tik (iterative)')
# plt.legend(loc=0, fontsize=12)
# plt.ylabel('MSE (Leanring Error)', fontsize=12)
# plt.xlabel('#of sample (Size of traing set)', fontsize=12)
# plt.title('%s user, %s alpha,%s g_lambda, %s noise'%(user_num, _lambda, g_lambda, noise_scale))
# plt.savefig(newpath+str(timeRun)+'_'+'mse_error_user_num_%s_noise_%s'%(user_num, noise_scale)+'.png', dpi=100)
# plt.show()



# plt.figure(figsize=(8,5))
# plt.plot(error_list_gb[3*user_num:], label='Graph-ridge Tik (iterative)')
# plt.legend(loc=0, fontsize=10)
# plt.ylabel('MSE (Leanring Error)', fontsize=12)
# plt.xlabel('#of sample (Size of traing set)', fontsize=12)
# plt.title('%s user, %s alpha,%s g_lambda, %s noise'%(user_num, _lambda, g_lambda, noise_scale))
# plt.savefig(newpath+str(timeRun)+'_'+'mse_error_user_num_%s_noise_%s_zoom_in'%(user_num, noise_scale)+'.png', dpi=100)	
# plt.show()


# graph, edge_num=create_networkx_graph(user_num, adj)
# edge_weight=adj[np.triu_indices(user_num,1)]
# edge_color=edge_weight[edge_weight>0]
# plt.figure(figsize=(5,5))
# nodes=nx.draw_networkx_nodes(graph, pos, node_size=50, cmap=plt.cm.jet)
# edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
# plt.axis('off')
# plt.savefig(newpath+str(timeRun)+'_'+'graph_user_num_%s_egde_num_%s'%(user_num, edge_num)+'.png', dpi=100)
# plt.show()


# for dim in range(dimension):
# 	plt.figure(figsize=(5,5))
# 	nodes=nx.draw_networkx_nodes(graph, pos, node_size=50, node_color=user_f[:,dim], cmap=plt.cm.jet)
# 	edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
# 	plt.axis('off')
# 	plt.savefig(newpath+str(timeRun)+'user_feature_smoothness_dimension_%s'%(dim)+'.png', dpi=100)
# 	plt.show()