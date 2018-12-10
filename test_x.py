import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import os 
os.chdir('../code/')
import datetime 
import networkx as nx
from utils import *
path='../results/'
user_num=10
item_num=50
dimension=10
noise_list=[0.01, 0.1, 0.25]
noise_scale=0.01
noise=np.random.normal(size=(user_num, item_num), scale=noise_scale)
_lambda=0.01
g_lambda=0.01
iteration=item_num
user_f,item_f,pos,ori_signal,adj,lap=generate_GMRF(user_num, item_num, dimension)
#lap=np.linalg.pinv(np.cov(user_f))
mask=np.zeros((user_num, item_num))
for i in range(user_num):
	mask[i, :i*np.int(item_num/user_num)]=1

if noise_scale==0:
	signal=ori_signal
else:
	signal=ori_signal+noise

beta_graph_ridge=np.zeros((user_num, dimension))
item_list=list(range(item_num))
error=np.zeros((item_num, user_num))
error_ols=np.zeros((item_num, user_num))
e_list1=[]
e_list2=[]
loss1=[]
for i in range(item_num):
	print('i/item_num', i, item_num)
	print('noise scale', noise_scale)
	X=item_f[:,item_list[:i+1]].T
	y=signal[:, item_list[:i+1]]
	mk=mask[:,:(i+1)]
	beta_graph_ridge=graph_ridge_no_mask(user_num, dimension, lap, X, y, g_lambda, 0)
	beta_ols=ridge_no_mask(user_num, dimension, lap, X, y, _lambda)
	e1=np.linalg.norm(beta_graph_ridge-user_f)
	e2=np.linalg.norm(beta_ols-user_f)
	e_list1.extend([e1])
	e_list2.extend([e2])
	for j in range(user_num):
		print('j', j)
		error[i,j]=np.linalg.norm(beta_graph_ridge[j]-user_f[j])
		error_ols[i,j]=np.linalg.norm(beta_ols[j]-user_f[j])


f, (ax1, ax2) = plt.subplots(1, 2, sharey=False, figsize=(10,10))
for i in range(user_num):
	ax1.plot(error[:,i],'.-', label='%s'%(i))
	ax2.plot(error_ols[:,i],'.-', label='%s'%(i))

ax1.set_title('graph ridge')
ax2.set_title('ols')
ax1.legend(loc=0)
ax2.legend(loc=2)
plt.savefig(path+'ind_error_no_mask_noise_%s'%(noise_scale)+'.png', dpi=100)
plt.show()

plt.figure(figsize=(15,10))
plt.plot(e_list1, label='graph_ridge')
plt.plot(e_list2, label='ols')
plt.legend(loc=0)
plt.savefig(path+'total_error_no_mask_noise_%s'%(noise_scale)+'.png', dpi=100)
plt.show()




graph, edge_num=create_networkx_graph(user_num, adj)
edge_weight=adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
plt.figure(figsize=(5,5))
nodes=nx.draw_networkx_nodes(graph, pos, node_size=50, cmap=plt.cm.jet)
edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
plt.axis('off')
plt.savefig(path+'user_graph_user_num_%s'%(user_num)+'.png', dpi=100)
plt.show()



