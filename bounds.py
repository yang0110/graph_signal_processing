import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge
from utils import create_networkx_graph
from sklearn import datasets
path='../results/Bound/'

np.random.seed(2019)

def lambda_(noise_list, d, user_num, dimension, item_num):
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
	bound=lam*np.sqrt(r)*(1+2*lap_user_infty)/(k+lam*lap_min)
	return bound 


user_num=200
dimension=5
item_num=200
d=3

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
Sigma=np.cov(item_f.T)
u,s,v=np.linalg.svd(Sigma)
sigma_min=np.min(s)
k=sigma_min/18

user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)
rank=np.linalg.matrix_rank(user_f)


ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
thrs_list=np.round(np.linspace((min_adj+max_adj)/2, max_adj, 5), decimals=4)
adj=ori_adj.copy()
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)
lap_user_fro=np.linalg.norm(np.dot(lap, user_f), 'fro')
lap_user_infty=np.linalg.norm(np.dot(lap, user_f), np.inf)

I=np.identity(user_num)
I_ev, I_evc=np.linalg.eig(I)
I_min=np.max(I_ev)
I_user_fro=np.linalg.norm(np.dot(I, user_f), 'fro')
I_user_infty=np.linalg.norm(np.dot(I, user_f), np.inf)

noise_list=[0.01,0.1,0.5, 1, 5 10]

noise_level=5

lam_list=[0.01,0.03,0.05,0.1,0.25, 0.5, 1]
graph_ridge_bound_infty_array=np.zeros((len(lam_list),item_num))
for index, lam in enumerate(lam_list):
	print('lam', lam)
	for i in range(item_num):
		print('i', i)
		x=item_f[:i+dimension,:]
		Sigma=np.cov(x.T)
		u,s,v=np.linalg.svd(Sigma)
		sigma_min=np.min(s)
		k=sigma_min/18
		graph_ridge_bound_infty_array[index, i]=graph_ridge_bound_infty(lam,rank, lap_user_infty, lap_min, k)

plt.figure()
for index, lam in enumerate(lam_list):
	plt.plot(graph_ridge_bound_infty_array[index], label=str(lam))

plt.legend(loc=0, fontsize=12)
plt.ylabel('Theoretical Bound', fontsize=12)
plt.xlabel('Sample size', fontsize=12)
plt.title('graph_ridge infty fixed lambda \n user num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
plt.savefig(path+'graph_ridge_infty_fixed_lam_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()


clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise

graph_ridge_error_array=np.zeros((len(lam_list), item_num))
for index, lam in enumerate(lam_list):
	print('lam', lam)
	for i in range(item_num):
		print('i', i)
		x=item_f[:i+dimension,:]
		y=noisy_signal[:, :i+dimension]
		A=lam*lap
		B=np.dot(x.T, x)
		C=np.dot(y, x)
		graph_ridge=scipy.linalg.solve_sylvester(A, B, C)
		graph_ridge_error_array[index, i]=np.linalg.norm(graph_ridge-user_f, 'fro')

plt.figure()
for index, lam in enumerate(lam_list):
	plt.plot(graph_ridge_error_array[index], label='lambda='+str(lam))

plt.legend(loc=0, fontsize=12)
plt.ylabel('Emperical Bound', fontsize=12)
plt.xlabel('Sample size', fontsize=12)
plt.title('graph_ridge error fixed lambda \n user num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
plt.savefig(path+'graph_ridge_empircial_error_fixed_lam_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()