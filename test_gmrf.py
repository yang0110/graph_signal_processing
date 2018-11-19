import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import datetime 
import networkx as nx

def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix

def generate_artificial_graph(user_num):
	pos=np.random.uniform(size=(user_num, 2))
	adj=rbf_kernel(pos)
	adj[adj<0.75]=0
	np.fill_diagonal(adj,0)
	lap=csgraph.laplacian(adj, normed=True)
	lap=normalized_trace(lap, user_num)
	return adj, lap, pos


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

def generate_random_graph(user_num, item_num, dimension, noise_scale):
	adj, lap, pos=generate_artificial_graph(user_num)
	b=np.random.uniform(size=(dimension, 2))
	user_f=np.dot(pos, b.T)
	user_f=Normalizer().fit_transform(user_f)
	item_f=np.random.uniform(size=(item_num, dimension))
	itme_f=Normalizer().fit_transform(item_f)
	signal=np.dot(user_f, item_f.T)
	noisy_signal=signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)
	return user_f, item_f.T, pos, noisy_signal, adj, lap 



def generate_GMRF(user_num, item_num, dimension, noise_scale):
	adj, lap, pos=generate_artificial_graph(user_num)
	cov=np.linalg.pinv(lap)+np.identity(user_num)
	# cov=np.linalg.pinv(lap)
	user_f=np.random.multivariate_normal(mean=np.zeros(user_num), cov=cov, size=dimension).T
	item_f=np.random.uniform(size=(item_num, dimension))
	itme_f=Normalizer().fit_transform(item_f)
	signal=np.dot(user_f, item_f.T)
	noisy_signal=signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)	
	return user_f, item_f.T, pos, noisy_signal, adj, lap

user_num=100
dimension=5
item_num=100
noise_scale=0.1
user_f, item_f, pos, noisy_signal, adj, lap=generate_GMRF(user_num, item_num, dimension, noise_scale)
# user_f, item_f, pos, noisy_signal, adj, lap=generate_random_graph(user_num, item_num, dimension, noise_scale)
graph, edge_num=create_networkx_graph(user_num, adj)
edge_weight=adj[np.triu_indices(user_num,1)]
edge_color=edge_weight[edge_weight>0]
for dim in range(dimension):
	plt.figure(figsize=(5,5))
	nodes=nx.draw_networkx_nodes(graph, pos, node_size=50, node_color=user_f[:,dim], cmap=plt.cm.jet)
	edges=nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.1, edge_color=edge_color, edge_cmap=plt.cm.Blues)
	plt.axis('off')
	plt.show()







