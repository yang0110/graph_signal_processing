import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import datetime 
import networkx as nx
from sklearn.datasets.samples_generator import make_blobs


def ols(x,y):
	cov=np.dot(x.T,x)
	temp2=np.linalg.inv(cov)
	beta=np.dot(np.dot(y,x), temp2)
	return beta 

# def ridge(x,y, _lambda, dimension):
# 	cov=np.dot(x.T,x)
# 	temp1=cov+_lambda*np.identity(dimension)
# 	temp2=np.linalg.inv(temp1)
# 	beta=np.dot(np.dot(y,x), temp2)
# 	return beta 

# Tikhonov
def graph_ridge(user_num, item_num, dimension, lap, item_f, mask, noisy_signal, alpha):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.multiply(cp.matmul(u, item_f),mask)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	#cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol 



def graph_ridge_iterative_model(item_f, noisy_signal, lap, beta_gb, dimension, user_list, item_list, user_dict, g_lambda):
	user_nb=len(user_list)
	item_nb=len(item_list)
	mask=np.ones((user_nb, item_nb))
	for ind, j in enumerate(user_list):
		remove_list=[x for x, xx in enumerate(item_list) if xx not in user_dict[j]]
		mask[ind, remove_list]=0
	signal=noisy_signal[user_list][:, item_list]
	signal=signal*mask
	sub_lap=lap[user_list][:,user_list]
	u=beta_gb[user_list].copy()
	sub_item_f=item_f[:, item_list]
	#update each atom
	for k in range(dimension):
		P=np.ones((user_nb, item_nb))
		P=P*(sub_item_f[k]!=0)
		p=P[0].reshape((1,item_nb))
		_sum=np.zeros((user_nb, item_nb))
		for kk in range(dimension):
			if kk==k:
				pass 
			else:
				_sum+=np.dot(u[:,kk].reshape((user_nb, 1)), sub_item_f[kk,:].reshape((1, item_nb)))
		e=signal-_sum*mask
		ep=e*P
		v_r=(sub_item_f[k,:].reshape((1,item_nb))*p).T 
		temp1=np.linalg.inv(np.dot(v_r.T, v_r)*np.identity(user_nb)+g_lambda*sub_lap)
		temp2=np.dot(ep, v_r)
		u[:,k]=np.dot(temp1, temp2).ravel()
	beta_gb[user_list]=u.copy()
	return beta_gb


def graph_ridge_mask(user_num, dimension, lap, item_f, noisy_signal, alpha, alpha2, mask):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.multiply(cp.matmul(u,item_f.T), mask)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	reg2=cp.sum([cp.quad_form(u[:,d], np.identity(user_num)) for d in range(dimension)])
	#cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp2=cp.Parameter(nonneg=True)
	alp.value=alpha 
	alp2.value=alpha2
	problem=cp.Problem(cp.Minimize(loss+alp*reg+alp2*reg2))
	problem.solve()
	sol=u.value 
	return sol 

def graph_ridge_no_mask(user_num, dimension, lap, item_f, noisy_signal, alpha, alpha2):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.matmul(u,item_f.T)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	reg2=cp.sum([cp.quad_form(u[:,d], np.identity(user_num)) for d in range(dimension)])
	#cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp2=cp.Parameter(nonneg=True)
	alp.value=alpha 
	alp2.value=alpha2
	problem=cp.Problem(cp.Minimize(loss+alp*reg+alp2*reg2))
	problem.solve()
	sol=u.value 
	return sol 

def ridge_mask(user_num, dimension, lap, item_f, noisy_signal, alpha, mask):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.multiply(cp.matmul(u,item_f.T), mask)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	lap=np.identity(user_num)
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	#cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol 

def ridge_no_mask(user_num, dimension, lap, item_f, noisy_signal, alpha):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.matmul(u,item_f.T)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	lap=np.identity(user_num)
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	#cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol 

def graph_ridge_iterative_model_no_mask(user_num, item_num, item_f, noisy_signal, lap, beta_gb, dimension, g_lambda):
	u=beta_gb.copy()
	item_nb=len(item_f)
	#update each atom
	for k in range(dimension):
		P=np.ones((user_num, item_nb))
		P=P*(item_f.T[k]!=0)
		p=P[0].reshape((1,item_nb))
		_sum=np.zeros((user_num, item_nb))
		for kk in range(dimension):
			if kk==k:
				pass 
			else:
				_sum+=np.dot(u[:,kk].reshape((user_num, 1)), item_f.T[kk,:].reshape((1, item_nb)))
		e=noisy_signal-_sum
		ep=e*P
		v_r=(item_f.T[k,:].reshape((1,item_nb))*p).T 
		temp1=np.linalg.inv(np.dot(v_r.T, v_r)*np.identity(user_num)+g_lambda*lap)
		temp2=np.dot(ep, v_r)
		u[:,k]=np.dot(temp1, temp2).ravel()
	beta_gb=u.copy()
	return beta_gb
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
	#pos=np.random.uniform(size=(user_num, 2))
	pos, y = make_blobs(n_samples=user_num, centers=5, n_features=2, random_state=0)
	pos=Normalizer().fit_transform(pos)
	adj=rbf_kernel(pos)
	#adj[adj<0.75]=0
	np.fill_diagonal(adj,0)
	lap=csgraph.laplacian(adj, normed=False)
	#lap=normalized_trace(lap, user_num)
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

def generate_random_graph(user_num, item_num, dimension):
	adj, lap, pos=generate_artificial_graph(user_num)
	b=np.random.uniform(size=(dimension, 2))
	user_f=np.dot(pos, b.T)
	user_f=Normalizer().fit_transform(user_f)
	item_f=np.random.uniform(size=(item_num, dimension))
	itme_f=Normalizer().fit_transform(item_f)
	signal=np.dot(user_f, item_f.T)
	return user_f, item_f.T, pos, signal, adj, lap 



def generate_GMRF(user_num, item_num, dimension):
	adj, lap, pos=generate_artificial_graph(user_num)
	cov=np.linalg.pinv(lap)
	user_f=np.random.multivariate_normal(mean=np.zeros(user_num), cov=cov, size=dimension).T
	item_f=np.random.uniform(size=(item_num, dimension))
	signal=np.dot(user_f, item_f.T)
	return user_f, item_f.T, pos, signal, adj, lap



def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix





# item_list=list(np.unique(item_list))
# user_nb=len(user_list)
# print('served user number', user_nb)
# item_nb=len(item_list)
# signal=noisy_signal[user_list][:,item_list]
# mask=np.ones((user_nb, item_nb))
# for ind, j in enumerate(user_list):
# 	remove_list=[x for x, xx in enumerate(item_list) if xx not in user_dict[j]]
# 	mask[ind, remove_list]=0
# signal=signal*mask
# sub_lap=lap[user_list][:,user_list]
# print('sub_lap.size', sub_lap.shape)
# u=beta_gb[user_list].copy()
# sub_item_f=item_f[:, item_list]
# #update each atom
# for k in range(dimension):
# 	P=np.ones((user_nb, item_nb))
# 	P=P*(sub_item_f[k]!=0)
# 	p=P[0].reshape((1,item_nb))
# 	_sum=np.zeros((user_nb, item_nb))
# 	for kk in range(dimension):
# 		if kk==k:
# 			pass 
# 		else:
# 			_sum+=np.dot(u[:,kk].reshape((user_nb, 1)), sub_item_f[kk,:].reshape((1, item_nb)))
# 	e=signal-_sum*mask
# 	ep=e*P
# 	v_r=(sub_item_f[k,:].reshape((1,item_nb))*p).T 
# 	temp1=np.linalg.inv(np.dot(v_r.T, v_r)*np.identity(user_nb)+g_lambda*sub_lap)
# 	temp2=np.dot(ep, v_r)
# 	u[:,k]=np.dot(temp1, temp2).ravel()
# beta_gb[user_list]=u.copy()
# error_gb=np.linalg.norm(beta_gb-user_f)
# error_list_gb.extend([error_gb])