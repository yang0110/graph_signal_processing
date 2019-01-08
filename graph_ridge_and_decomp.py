import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
# os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from sklearn import datasets
#from utils import *
path='../results/'

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

user_num=50
item_num=100
dimension=25
noise_level=0.5
lam=0.01

user_f=np.random.normal(size=(user_num, dimension))
user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
lap=csgraph.laplacian(ori_adj, normed=False)
lap_evalues, lap_evectors=np.linalg.eig(lap)

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal

graph_sly_array=np.zeros(item_num)
graph_decomp_array=np.zeros(item_num)
ridge_decomp_array=np.zeros(item_num)
diff=np.zeros(item_num)
graph_sly_error_array=np.zeros((user_num, item_num))
graph_decomp_error_array=np.zeros((user_num, item_num))
ridge_decomp_error_array=np.zeros((user_num, item_num))

for i in range(item_num):
	print('i', i)
	x=item_f[:i+1,:]
	y=noisy_signal[:, :i+1]
	A=lam*lap 
	B=np.dot(x.T,x)
	C=np.dot(y,x)
	graph_sly_res=scipy.linalg.solve_sylvester(A,B,C)
	graph_sly_array[i]=np.linalg.norm(graph_sly_res-user_f, 'fro')
	graph_decomp_res=graph_ridge_decomp(user_num, dimension, x, y, lam, lap_evalues)
	graph_decomp_array[i]=np.linalg.norm(graph_decomp_res-user_f, 'fro')
	ridge_decomp_res=ridge_decomp(user_num,dimension, x,y, lam)
	ridge_decomp_array[i]=np.linalg.norm(ridge_decomp_res-user_f, 'fro')
	diff[i]=np.linalg.norm(graph_sly_res-graph_decomp_res)
	graph_sly_error_array[:, i]=np.linalg.norm(graph_sly_res-user_f, axis=1)
	graph_decomp_error_array[:, i]=np.linalg.norm(graph_decomp_res-user_f, axis=1)
	ridge_decomp_error_array[:,i]=np.linalg.norm(ridge_decomp_res-user_f, axis=1)

plt.figure()
plt.plot(ridge_decomp_array, label='ridge')
plt.plot(graph_sly_array, 'k+-', markevery=0.05, label='graph-sly')
plt.plot(graph_decomp_array, 'ro-', markevery=0.1, label='graph-decomp')
plt.legend(loc=0, fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.xlabel('Sample size', fontsize=12)
plt.savefig(path+'graph_ridge_and_decomp_and_ridge'+'.png', dpi=300)
plt.show()

# labels=['1','2','3','4','5']
# fig, (ax1, ax2)=plt.subplots(1,2)
# ax1.plot(graph_sly_error_array[:5,:].T)
# ax1.set_title('graph sly', fontsize=12)
# ax1.set_ylim([0,1])
# ax1.legend(loc=0, labels=labels)
# ax2.plot(graph_decomp_error_array[:5,:].T)
# ax2.set_title('graph decomp', fontsize=12)
# ax2.legend(loc=0, labels=labels)
# ax2.set_ylim([0,1])
# plt.show()


# labels=['1','2','3','4','5']
# fig, (ax1, ax2)=plt.subplots(1,2)
# ax1.plot(ridge_decomp_error_array[:5,:].T)
# ax1.set_title('ridge decomp', fontsize=12)
# ax1.legend(loc=0, labels=labels)
# ax1.set_ylim([0,1])
# ax2.plot(graph_decomp_error_array[:5,:].T)
# ax2.set_title('graph decomp', fontsize=12)
# ax2.legend(loc=0, labels=labels)
# ax2.set_ylim([0,1])
# plt.show()

lap_evalues_matrix=np.diag(lap_evalues)
A=np.dot(lap_evectors, np.sqrt(np.linalg.inv(lap_evalues_matrix)))
A_inv=np.linalg.inv(A)
norm1=np.linalg.norm(np.dot(A_inv, user_f)[1])**2
norm2=lap_evalues[-1]*(np.linalg.norm(user_f[1])**2)

norm1=np.trace(np.dot(np.dot(user_f.T, lap), user_f))
norm2=np.sum(lap_evalues*(np.linalg.norm(user_f)**2))
norm3=np.linalg.norm(np.dot(A_inv, user_f),'fro')**2





