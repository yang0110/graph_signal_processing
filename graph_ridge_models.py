import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import os 
os.chdir('/media/kaige/New SSD/Research/Code/GSP_code/')
import datetime 
import networkx as nx
from utils import *


timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 
newpath='/media/kaige/New SSD/Research/Results/linear_signal_results/'
if not os.path.exists(newpath):
	os.makedirs(newpath)


user_num=10
item_num=100
dimension=10
noise_scale=0.25

user_f,item_f,noisy_signal,adj,lap=generate_graph_and_atom(user_num, item_num, dimension, noise_scale)

_lambda=0.1
g_lambda=0.1
iteration=1000

all_user=list(range(user_num))
all_item=list(range(item_num))
user_list=[]
item_list=[]

beta_ols=np.zeros((user_num, dimension))
beta_ridge=np.zeros((user_num, dimension))
beta_graph_ridge=np.zeros((user_num, dimension))
beta_gb=np.zeros((user_num, dimension))
beta_graph_ridge_weighted=np.zeros((user_num, dimension))
weights=np.zeros(user_num)
user_dict={a:[] for a in all_user}
error_list_ols=[]
error_list_ridge=[]
error_list_graph_ridge=[]
error_list_gb=[]
error_list_graph_ridge_weighted=[]



for i in range(iteration):
	print('iteration i=', i )
	if i<user_num:
		user=all_user[i]
	else:
		user=np.random.choice(all_user)
	item=np.random.choice(all_item)
	user_dict[user].extend([item])
	item_sub_list=user_dict[user]
	X=item_f[:, item_sub_list].T
	y=noisy_signal[user, item_sub_list]
	### OLS
	beta_ols[user]=ridge(X, y, 0.01, dimension)
	error_ols=np.linalg.norm(beta_ols-user_f)
	error_list_ols.extend([error_ols])
	### Ridge 
	beta_ridge[user]=ridge(X, y, _lambda, dimension)
	error_ridge=np.linalg.norm(beta_ridge-user_f)
	error_list_ridge.extend([error_ridge])
	### Graph Ridge 
	user_list.extend([user])
	item_list.extend([item])
	user_list=list(np.unique(user_list))
	item_list=list(np.unique(item_list))
	user_nb=len(user_list)
	print('served user number', user_nb)
	item_nb=len(item_list)
	item_feature=item_f[:, item_list]
	signal=noisy_signal[user_list][:,item_list]
	mask=np.ones((user_nb, item_nb))
	for ind, j in enumerate(user_list):
		remove_list=[x for x, xx in enumerate(item_list) if xx not in user_dict[j]]
		mask[ind, remove_list]=0
	signal=signal*mask
	sub_lap=lap[user_list][:,user_list]
	print('sub_lap.size', sub_lap.shape)
	u_f=graph_ridge(user_nb, item_nb, dimension, sub_lap, item_feature, mask, signal, g_lambda)
	beta_graph_ridge[user_list]=u_f.copy()
	error_graph_ridge=np.linalg.norm(beta_graph_ridge-user_f)
	error_list_graph_ridge.extend([error_graph_ridge])
	P=np.ones((user_nb, item_nb))
	u=np.zeros((user_nb, dimension))
	sub_item_f=item_f[:, item_list]
	for k in range(dimension):
		P=P*(sub_item_f[k]!=0)
		p=P[0].reshape((1,item_nb))
		# update each atom
		_sum=np.zeros((user_nb, item_nb))
		for kk in range(dimension):
			if kk==k:
				pass 
			else:
				_sum+=np.dot(u[:,kk].reshape((user_nb, 1)), sub_item_f[kk,:].reshape((1, item_nb)))*mask
			e=signal-_sum 
		ep=e*P
		v_r=(sub_item_f[k,:].reshape((1,item_nb))*p).T 
		temp1=np.linalg.inv(np.linalg.norm(v_r)*np.identity(user_nb)+g_lambda*sub_lap)
		temp2=np.dot(ep, v_r)
		u[:,k]=np.dot(temp1, temp2).ravel()
	beta_gb[user_list]=u.copy()
	error_gb=np.linalg.norm(beta_gb-user_f)
	error_list_gb.extend([error_gb])
	# Graph ridge weighted
	weights[user]=1/np.trace(noise_scale*np.linalg.inv(np.dot(X.T,X)+0.01*np.identity(dimension)))
	sub_weights=np.diag(weights[user_list])
	gb_weight=graph_ridge_weighted(user_nb, item_nb, dimension, sub_weights, sub_lap, item_feature, mask, signal, g_lambda)
	beta_graph_ridge_weighted[user_list]=gb_weight.copy()
	error_graph_ridge_weighted=np.linalg.norm(beta_graph_ridge_weighted-user_f)
	error_list_graph_ridge_weighted.extend([error_graph_ridge_weighted])


plt.figure(figsize=(8,5))
plt.plot(error_list_ols, label='OLS')
plt.plot(error_list_ridge, label='Ridge')
plt.plot(error_list_graph_ridge,marker='s', markevery=0.1,label='Graph-ridge TiK (convex)')
plt.plot(error_list_gb, label='Graph-ridge TiK (iterative)')
plt.plot(error_list_graph_ridge_weighted, marker='*', markevery=0.1, label='Graph_ridge_weighted Tik (convex)')
plt.legend(loc=0, fontsize=12)
plt.ylabel('MSE (Leanring Error)', fontsize=12)
plt.xlabel('#of sample (Size of traing set)', fontsize=12)
plt.title('%s user, %s alpha,%s g_lambda, %s noise'%(user_num, _lambda, g_lambda, noise_scale))
plt.savefig(newpath+str(timeRun)+'_'+'mse_error'+'.png', dpi=300)
plt.show()




plt.figure(figsize=(8,5))
plt.plot(error_list_ols[3*user_num:], label='OLS')
plt.plot(error_list_ridge[3*user_num:], label='Ridge')
plt.plot(error_list_graph_ridge[3*user_num:], marker='s',markevery=0.1, label='Graph-ridge Tik (convex)')
plt.plot(error_list_gb[3*user_num:], label='Graph-ridge Tik (iterative)')
plt.plot(error_list_graph_ridge_weighted[3*user_num:], marker='*', markevery=0.1, label='Graph-ridge-weighted Tik  (convex)')
plt.legend(loc=0, fontsize=10)
plt.ylabel('MSE (Leanring Error)', fontsize=12)
plt.xlabel('#of sample (Size of traing set)', fontsize=12)
plt.title('%s user, %s alpha,%s g_lambda, %s noise'%(user_num, _lambda, g_lambda, noise_scale))
plt.savefig(newpath+str(timeRun)+'_'+'mse_error_zoom_in'+'.png', dpi=300)
plt.show()
