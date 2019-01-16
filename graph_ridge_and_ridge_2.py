import numpy as np 
import cvxpy as cp 
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from utils import *
from sklearn import datasets
path='../results/Graph_ridge_results/'

def graph_ridge_decomp(user_num, dimension, X, Y, lam, lap_evalues):
	user_est=np.zeros((user_num, dimension))
	I=np.identity(dimension)
	for t in range(user_num):
		print('user', t)
		lam_t=lam*lap_evalues[t] 
		y=Y[t].reshape((1, len(Y[t])))
		a=np.dot(y, X)
		b=np.linalg.pinv(np.dot(X.T, X)+lam_t*I)
		user_est[t]=np.dot(a,b)
	return user_est 

np.random.seed(seed=2019)

user_num=50
item_num=200
dimension=10
noise_level=0.75
lam=0.01

I=np.identity(user_num)
user_f=np.random.normal(size=(user_num, dimension))
#user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
adj=ori_adj.copy()
thrs=(min_adj+max_adj)/2
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)
lap_evalues, lap_evectors=np.linalg.eig(lap)
lap_evalues=np.sort(lap_evalues)
##generate item_f
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

## signal 
clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 

ridge_error_array=np.zeros(item_num)
graph_ridge_error_array=np.zeros(item_num)
graph_ridge_decomp_array=np.zeros(item_num)
for i in range(item_num):
	print('i, lam', i, lam)
	x=item_f[:i+dimension, :]
	y=noisy_signal[:, :i+dimension]
	A=lam*lap
	B=np.dot(x.T, x)
	C=np.dot(y,x)
	graph_ridge=scipy.linalg.solve_sylvester(A, B, C)
	graph_ridge_error_array[i]=np.linalg.norm(graph_ridge-user_f, 'fro')
	AA=lam*I 
	ridge=scipy.linalg.solve_sylvester(AA, B, C)
	ridge_error_array[i]=np.linalg.norm(ridge-user_f, 'fro')
	graph_ridge_decomp_res=graph_ridge_decomp(user_num, dimension, x, y, lam, lap_evalues)
	graph_ridge_decomp_array[i]=np.linalg.norm(graph_ridge_decomp_res-user_f, 'fro')



plt.figure()
plt.plot(ridge_error_array, 'r+-', markevery=0.1,label='ridge')
plt.plot(graph_ridge_error_array,'b', label='graph ridge')
plt.plot(graph_ridge_decomp_array, 'y', label='graph decomp')
plt.legend(loc=0, fontsize=12)
plt.xlabel('Sample size', fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.title('user num=%s, noise=%s'%(user_num, noise_level), fontsize=12)
plt.savefig(path+'no_mask_graph_ridge_vs_ridge_user_num_%s_noise_%s'%(user_num, noise_level)+'.png', dpi=300)
plt.show()



