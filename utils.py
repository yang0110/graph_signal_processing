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
	graph=nx.barabasi_albert_graph(node_num, m=1, seed=seed)
	adj_matrix=nx.to_numpy_array(graph)
	np.fill_diagonal(adj_matrix,0)
	lap=csgraph.laplacian(adj_matrix, normed=False)
	return adj_matrix, lap

def generate_graph_and_atom(user_num, item_num, dimension, noise_scale):
	adj, lap=generate_ba_graph(user_num)
	p_lap=np.linalg.pinv(lap)
	cov=np.linalg.pinv(lap)+np.identity(user_num)
	U=np.random.multivariate_normal(mean=np.zeros(user_num), cov=cov, size=dimension).T
	item_f=np.random.normal(size=(dimension, item_num))# K*M
	item_f=Normalizer().fit_transform(item_f)
	U=Normalizer().fit_transform(U)
	signal=np.dot(U, item_f)# N*M 
	noisy_signal=signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)
	return U, item_f, noisy_signal, adj, lap 
