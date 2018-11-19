import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import datetime 
import networkx as nx


def ols(x,y):
	cov=np.dot(x.T,x)
	temp2=np.linalg.inv(cov)
	beta=np.dot(temp2, np.dot(x.T,y)).T
	return beta 

def ridge(x,y, _lambda, dimension):
	cov=np.dot(x.T,x)
	temp1=cov+_lambda*np.identity(dimension)
	temp2=np.linalg.inv(temp1)
	beta=np.dot(temp2, np.dot(x.T,y)).T
	return beta 

# Tikhonov
def graph_ridge(user_num, item_num, dimension, lap, item_f, mask, noisy_signal, alpha):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.multiply(cp.matmul(u, item_f),mask)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol 

# Total variance
def graph_ridge_TV(user_num, item_num, dimension, lap, item_f, mask, noisy_signal, alpha):
	lap=np.sqrt(lap)
	u=cp.Variable((user_num, dimension))
	l_signal=cp.multiply(cp.matmul(u, item_f),mask)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol 

def graph_ridge_weighted(user_num, item_num, dimension, weights, lap, item_f, mask, noisy_signal, alpha):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.multiply(cp.matmul(u, item_f), mask)
	loss1=noisy_signal-l_signal
	loss=cp.sum([cp.quad_form(loss1[:,dd], weights) for dd in range(item_num)])
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol 


def generate_data(user_num, item_num, dimension, noise_scale):
	user_f=np.random.normal(size=(user_num, dimension))# N*K
	user_f=Normalizer().fit_transform(user_f)
	item_f=np.random.normal(size=(dimension, item_num))# K*M
	item_f=Normalizer().fit_transform(item_f.T).T
	signal=np.dot(user_f, item_f)# N*M 
	noisy_signal=signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)
	true_adj=rbf_kernel(user_f)
	np.fill_diagonal(true_adj, 0)
	lap=csgraph.laplacian(true_adj, normed=False)
	return user_f, item_f, noisy_signal, true_adj, lap 
	
def generate_ba_graph(node_num, seed=2018):
	graph=nx.barabasi_albert_graph(node_num, m=5, seed=seed)
	adj_matrix=nx.to_numpy_array(graph)
	np.fill_diagonal(adj_matrix,0)
	lap=csgraph.laplacian(adj_matrix, normed=False)
	return adj_matrix, lap

def generate_erdos_renyi_graph(user_num, p):
	graph=nx.erdos_renyi_graph(user_num,p=p)
	adj=nx.to_numpy_array(graph)
	np.fill_diagonal(adj,0)
	return adj 



def generate_graph_and_atom(user_num, item_num, dimension, noise_scale, p):
	mask=generate_erdos_renyi_graph(user_num, p=p)
	mask=np.fmax(mask, mask.T)
	adj=np.random.uniform(size=(user_num, user_num))
	adj=np.fmax(adj, adj.T)
	adj=adj*mask
	lap=csgraph.laplacian(adj, normed=False)
	cov=np.linalg.pinv(lap)+np.identity(user_num)
	U=np.random.multivariate_normal(mean=np.zeros(user_num), cov=cov, size=dimension).T
	item_f=np.random.normal(size=(dimension, item_num))# K*M
	item_f=Normalizer().fit_transform(item_f)
	U=Normalizer().fit_transform(U)
	signal=np.dot(U, item_f)# N*M 
	noisy_signal=signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)
	return U, item_f, noisy_signal, adj, lap 




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
	cov=np.linalg.pinv(lap)+0.1*np.identity(user_num)
	cov=np.linalg.pinv(lap)
	user_f=np.random.multivariate_normal(mean=np.zeros(user_num), cov=cov, size=dimension).T
	item_f=np.random.uniform(size=(item_num, dimension))
	itme_f=Normalizer().fit_transform(item_f)
	signal=np.dot(user_f, item_f.T)
	noisy_signal=signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)	
	return user_f, item_f.T, pos, noisy_signal, adj, lap



def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix





