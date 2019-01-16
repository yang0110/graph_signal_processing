import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge
from utils import create_networkx_graph
from sklearn import datasets
path='../results/Bound/emperical bound/'

np.random.seed(2019)

def lambda_(noise_level, d, user_num, dimension, item_num):
	lam=8*np.sqrt(noise_level)*np.sqrt(d)*np.sqrt(user_num+dimension)/(user_num*item_num)
	return lam 

def ridge_bound_fro(lam, rank, I_user_fro, I_min, k):
	bound=lam*(np.sqrt(rank)+2*I_user_fro)/(k+lam*I_min)
	return bound 

def ridge_bound_infty(lam, rank, I_user_infty, I_min, k):
	bound=lam*np.sqrt(rank)*(1+2*I_user_infty)/(k+lam*I_min)
	return bound 

def graph_ridge_bound_fro(lam, rank, lap_user_fro, lap_min, k):
	bound=lam*(np.sqrt(rank)+2*lap_user_fro)/(k+lam*lap_min)
	return bound 

def graph_ridge_bound_infty(lam, rank, lap_user_infty, lap_min, k):
	bound=lam*np.sqrt(rank)*(1+2*lap_user_infty)/(k+lam*lap_min)
	return bound 


def graph_ridge_decomp(user_num, dimension, X, Y, lam, lap_evalues):
	user_est=np.zeros((user_num, dimension))
	I=np.identity(dimension)
	for t in range(user_num):
		lam_t=lam*lap_evalues[t] 
		y=Y[t]
		a=np.dot(y, X)
		b=np.linalg.pinv(np.dot(X.T, X)+lam_t*I)
		user_est[t]=np.dot(a,b)
	return user_est 

def ridge_decomp(user_num, dimension, X, Y, lam):
	user_est=np.zeros((user_num, dimension))
	I=np.identity(dimension)
	for t in range(user_num):
		y=Y[t]
		a=np.dot(y, X)
		b=np.linalg.pinv(np.dot(X.T, X)+lam*I)
		user_est[t]=np.dot(a,b)
	return user_est 

user_num=100
dimension=25
item_num=500
noise_level=0.5

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

user_f=np.random.normal(size=(user_num, dimension))
# user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)

adj=rbf_kernel(user_f)
lap=csgraph.laplacian(adj, normed=False)
I=np.identity(user_num)

clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise

graph_array=np.zeros(item_num)
ridge_array=np.zeros(item_num)

lam=1
lam2=lam*25
for i in range(item_num):
	print('i', i)
	x=item_f[:i+dimension,:]
	y=noisy_signal[:, :i+dimension]
	A=lam2*lap 
	AA=lam*I
	B=np.dot(x.T,x)
	C=np.dot(y,x)
	graph_res=scipy.linalg.solve_sylvester(A,B,C)
	graph_array[i]=np.linalg.norm(graph_res-user_f, 'fro')
	ridge_res=scipy.linalg.solve_sylvester(AA, B, C)
	ridge_array[i]=np.linalg.norm(ridge_res-user_f, 'fro')


plt.figure(figsize=(5,5))
plt.plot(ridge_array[dimension:], label='Ridge')
plt.plot(graph_array[dimension:], label='Graph-Ridge')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Sample size (m)', fontsize=12)
plt.ylabel('Emperical Error', fontsize=12)
plt.savefig(path+'01_lambda_ridge_vs_graph_ridge_noise_%s'%(noise_level)+'.png', dpi=200)
plt.show()














