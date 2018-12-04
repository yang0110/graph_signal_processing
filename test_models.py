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

timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 
newpath='/home/kaige/Documents/results/'

user_num=25
item_num=200
dimension=25
noise_list=[0.01, 0.1, 0.25]
loop_num=3
_lambda=0.1
g_lambda=0.1
iteration=item_num

user_f,item_f,pos,ori_signal,adj,lap=generate_GMRF(user_num, item_num, dimension)
# cov=np.cov(user_f)
# lap=cov
error_dict_ols={}
error_dict_ridge={}
error_dict_graph_ridge={}
error_dict_gb={}
item_list=list(range(item_num))
for loop in range(loop_num):
	for noise_scale in noise_list:
		print('noise scale=%s'%(noise_scale))
		noisy_signal=ori_signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)
		beta_ols=np.zeros((user_num, dimension))
		beta_ridge=np.zeros((user_num, dimension))
		beta_graph_ridge=np.zeros((user_num, dimension))
		beta_gb=np.zeros((user_num, dimension))
		error_list_ols=[]
		error_list_ridge=[]
		error_list_graph_ridge=[]
		error_list_gb=[]
		for i in range(iteration):
			print('loop/total loop', loop, loop_num)
			print('iteration i=', i )
			print('noise scale=', noise_scale)
			X=item_f[:,item_list[:i+1]].T
			y=noisy_signal[:, item_list[:i+1]]
			### OLS
			beta_ols=ridge(X, y, 0.01, dimension)
			error_ols=np.linalg.norm(beta_ols-user_f)
			error_list_ols.extend([error_ols])
			### Ridge 
			beta_ridge=ridge(X, y, _lambda, dimension)
			error_ridge=np.linalg.norm(beta_ridge-user_f)
			error_list_ridge.extend([error_ridge])
			### Graph Ridge 
			u_f=graph_ridge_no_mask(user_num, dimension, lap, X, y, g_lambda)
			beta_graph_ridge=u_f.copy()
			error_graph_ridge=np.linalg.norm(beta_graph_ridge-user_f)
			error_list_graph_ridge.extend([error_graph_ridge])
			# graph ridge iterative 
			beta_gb=graph_ridge_iterative_model_no_mask(user_num, item_num, X, y, lap, beta_gb, dimension, g_lambda)
			error_gb=np.linalg.norm(beta_gb-user_f)
			error_list_gb.extend([error_gb])
		if loop==0:
			error_dict_ols[noise_scale]=np.array(error_list_ols)
			error_dict_ridge[noise_scale]=np.array(error_list_ridge)
			error_dict_graph_ridge[noise_scale]=np.array(error_list_graph_ridge)
			error_dict_gb[noise_scale]=np.array(error_list_gb)
		else:
			error_dict_ols[noise_scale]=0.5*(np.array(error_list_ols)+error_dict_ols[noise_scale])
			error_dict_ridge[noise_scale]=0.5*(np.array(error_list_ridge)+error_dict_ridge[noise_scale])
			error_dict_graph_ridge[noise_scale]=0.5*(np.array(error_list_graph_ridge)+error_dict_graph_ridge[noise_scale])
			error_dict_gb[noise_scale]=0.5*(np.array(error_list_gb)+error_dict_gb[noise_scale])


for noise_scale in noise_list:
	plt.figure()
	plt.plot(error_dict_ols[noise_scale][user_num:], label='OLS')
	plt.plot(error_dict_ridge[noise_scale][user_num:], label='Ridge')
	plt.plot(error_dict_graph_ridge[noise_scale][user_num:], marker='s',markevery=0.1, label='Graph-ridge Tik (convex)')
	plt.plot(error_dict_gb[noise_scale][user_num:], label='Graph-ridge Tik (iterative)')
	plt.legend(loc=0, fontsize=10)
	plt.ylabel('MSE (Leanring Error)', fontsize=12)
	plt.xlabel('#of sample (Size of traing set)', fontsize=12)
	plt.savefig(newpath+str(timeRun)+'_'+'mse_error_user_num_%s_noise_%s_zoom_in'%(user_num, noise_scale)+'.png', dpi=100)	
	plt.clf()


plt.figure()
for noise_scale in noise_list:
	plt.plot(error_dict_ols[noise_scale][user_num:], label='Noise=%s'%(noise_scale))
plt.legend(loc=0)
plt.ylabel('MSE (Leanring Error)', fontsize=12)
plt.xlabel('#of sample (Size of traing set)', fontsize=12)
plt.savefig(newpath+str(timeRun)+'OLS_error_tune_noise_user_num_%s'%(user_num)+'.png', dpi=100)
plt.clf()



plt.figure()
for noise_scale in noise_list:
	plt.plot(error_dict_ridge[noise_scale][user_num:], label='Noise=%s'%(noise_scale))
plt.legend(loc=0)
plt.ylabel('MSE (Leanring Error)', fontsize=12)
plt.xlabel('#of sample (Size of traing set)', fontsize=12)
plt.savefig(newpath+str(timeRun)+'ridge_error_tune_noise_user_num_%s'%(user_num)+'.png', dpi=100)
plt.clf()



plt.figure()
for noise_scale in noise_list:
	plt.plot(error_dict_graph_ridge[noise_scale][user_num:], label='Noise=%s'%(noise_scale))
plt.legend(loc=0)
plt.ylabel('MSE (Leanring Error)', fontsize=12)
plt.xlabel('#of sample (Size of traing set)', fontsize=12)
plt.savefig(newpath+str(timeRun)+'graph_ridge_error_tune_noise_user_num_%s'%(user_num)+'.png', dpi=100)
plt.clf()



plt.figure()
for noise_scale in noise_list:
	plt.plot(error_dict_gb[noise_scale][user_num:], label='Noise=%s'%(noise_scale))
plt.legend(loc=0)
plt.ylabel('MSE (Leanring Error)', fontsize=12)
plt.xlabel('#of sample (Size of traing set)', fontsize=12)
plt.savefig(newpath+str(timeRun)+'graph_ridge_iterative_error_tune_noise_user_num_%s'%(user_num)+'.png', dpi=100)
plt.clf()