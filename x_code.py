import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer, MinMaxScaler
from scipy.sparse import csgraph 
import datetime 
import networkx as nx
from sklearn.datasets.samples_generator import make_blobs

def ols(x,y, dimension):
	cov=np.dot(x.T,x)
	temp2=np.linalg.inv(cov+0.01*np.identity(dimension))
	beta=np.dot(np.dot(y,x), temp2)
	return beta 

def graph_ridge_no_mask(user_num, dimension, lap, item_f, noisy_signal, alpha):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.matmul(u,item_f.T)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol 

def create_networkx_graph(node_num, adj_matrix):
	G=nx.Graph()
	G.add_nodes_from(list(range(node_num)))
	for i in range(node_num):
		for j in range(node_num):
			if adj_matrix[i,j]!=0:
				G.add_edge(i,j,weight=adj_matrix[i,j])
			else:
				pass
	edge_num=G.number_of_edges()
	return G, edge_num

def create_graph(user_num, cluster_num, cluster_std):
	pos, label=make_blobs(n_samples=user_num, centers=cluster_num,  n_features=2, cluster_std=cluster_std, center_box=(-1.0, 1.0), shuffle=False)
	adj=rbf_kernel(pos)
	np.fill_diagonal(adj,0)
	lap=csgraph.laplacian(adj, normed=False)
	return pos, adj, lap 

def create_graph_signal(user_num, lap, dimension):
	cov=np.linalg.pinv(lap)
	signals=np.random.multivariate_normal(mean=np.zeros(user_num), cov=cov, size=dimension).T	
	return signals

user_num=25
item_num=500
dimension=10
cluster_num=5
cluster_std=0.1
noise_scale=0.1
g_lambda=0.1

pos, adj, lap=create_graph(user_num, cluster_num, cluster_std)
signals=create_graph_signal(user_num, lap, dimension)

# graph, edge_num=create_networkx_graph(user_num, adj)
# edge_weight=adj[np.triu_indices(user_num,1)]
# edge_color=edge_weight[edge_weight>0]
# for dim in range(dimension):
# 	plt.figure(figsize=(5,5))
# 	nodes=nx.draw_networkx_nodes(graph, pos, node_size=50, node_color=signals[:,dim], cmap=plt.cm.jet)
# 	edges=nx.draw_networkx_edges(graph, pos, width=0.1, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
# 	plt.axis('off')
# 	plt.show()


item_f=np.random.multivariate_normal(mean=np.zeros(dimension), cov=np.identity(dimension), size=item_num).T

payoffs=np.dot(signals, item_f)
noisy_payoffs=payoffs+np.random.normal(size=(user_num, item_num), scale=noise_scale)

ols_er_list=[]
gb_er_list=[]
for i in range(item_num):
	print('i', i)
	x=item_f[:,:i+1].T
	y=noisy_payoffs[:,:i+1]
	ols_beta=ols(x, y, dimension)
	ols_er=np.linalg.norm(ols_beta-signals)
	ols_er_list.extend([ols_er])
	gb_beta=graph_ridge_no_mask(user_num, dimension, lap, x, y, g_lambda)
	gb_er=np.linalg.norm(gb_beta-signals)
	gb_er_list.extend([gb_er])


plt.figure()
plt.plot(gb_er_list, label='Graph ridge')
plt.plot(ols_er_list, label='OLS')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Training set size', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.show()
