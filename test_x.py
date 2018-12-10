import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import os 
os.chdir('/home/kaige/Documents/code/')
import datetime 
import networkx as nx
from utils import *
path='/home/kaige/Documents/results/'
user_num=5
item_num=100
dimension=10
noise_list=[0.01, 0.1, 0.25]
_lambda=0.01
g_lambda=0.01
iteration=item_num
user_f,item_f,pos,ori_signal,adj,lap=generate_GMRF(user_num, item_num, dimension)
#lap=np.linalg.pinv(np.cov(user_f))
mask=np.zeros((user_num, item_num))
for i in range(user_num):
	mask[i, :i*np.int(item_num/user_num)]=1

signal=ori_signal*mask

beta_graph_ridge=np.zeros((user_num, dimension))
item_list=list(range(item_num))
error=np.zeros((item_num, user_num))
error_ols=np.zeros((item_num, user_num))
e_list1=[]
e_list2=[]
for i in range(item_num):
	print('i/item_num', i, item_num)
	X=item_f[:,item_list[:i+1]].T
	y=signal[:, item_list[:i+1]]
	mk=mask[:,:(i+1)]
	beta_graph_ridge=graph_ridge_mask(user_num, dimension, lap, X, y, g_lambda, 0, mk)
	beta_ols=ridge_mask(user_num, dimension, lap, X, y, _lambda,mk)
	e1=np.linalg.norm(beta_graph_ridge-user_f)
	e2=np.linalg.norm(beta_ols-user_f)
	e_list1.extend([e1])
	e_list2.extend([e2])
	for j in range(user_num):
		print('j', j)
		error[i,j]=np.linalg.norm(beta_graph_ridge[j]-user_f[j])
		error_ols[i,j]=np.linalg.norm(beta_ols[j]-user_f[j])


f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
for i in range(user_num):
	ax1.plot(error[:,i],'.-', label='%s'%(i))
	ax2.plot(error_ols[:,i],'.-', label='%s'%(i))

ax1.set_title('graph ridge')
ax2.set_title('ols')
ax1.legend(loc=0)
ax2.legend(loc=2)
plt.show()

plt.figure()
plt.plot(e_list1, label='graph_ridge')
plt.plot(e_list2, label='ols')
plt.legend(loc=0)
plt.show()


ee=[]
for d in range(5, 100):
	dimension=d
	user_f,item_f,pos,ori_signal,adj,lap=generate_GMRF(user_num, item_num, dimension)
	cov=np.cov(user_f)
	sigma=np.linalg.pinv(lap)
	ee.extend([np.linalg.norm(cov-sigma)])

plt.figure()
plt.plot(ee)
plt.xlabel('K', fontsize=12)
plt.ylabel('MSE', fontsize=12)
plt.savefig(path+'mse_sample_cov_lap'+'.png', dpi=10)
plt.show()