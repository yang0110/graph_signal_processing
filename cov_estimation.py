import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import datetime 
import networkx as nx
def generate_artificial_graph(user_num):
	pos=np.random.uniform(size=(user_num, 2))
	adj=rbf_kernel(pos)
	# adj[adj<0.75]=0
	np.fill_diagonal(adj,0)
	lap=csgraph.laplacian(adj, normed=False)
	lap=normalized_trace(lap, user_num)
	return adj, lap, pos

def normalized_trace(matrix, target_trace):
	normed_matrix=target_trace*matrix/np.trace(matrix)
	return normed_matrix

def generate_GMRF(user_num,dimension, noise_scale):
	adj, lap, pos=generate_artificial_graph(user_num)
	cov=np.linalg.pinv(lap)
	user_f=np.random.multivariate_normal(mean=np.zeros(user_num), cov=cov, size=dimension).T
	#user_f=Normalizer().fit_transform(user_f)
	if noise_scale==0:
		pass 
	else:
		noise=np.random.normal(size=(user_num, dimension), scale=noise_scale)
		user_f=user_f+noise
	return user_f, cov


user_num=10
dimension=5
dimension_list=np.arange(5, 1000, 1)
noise_scale=0
error_list=[]
for dimension in dimension_list:
	U, cov_true=generate_GMRF(user_num, dimension, noise_scale)
	cov=np.cov(U)
	error=np.linalg.norm(cov_true-cov)
	error_list.extend([error])

plt.figure()
plt.plot(dimension_list, error_list)
plt.xlabel('sample number')
plt.ylabel('error')
plt.show()
