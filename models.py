import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import os 
os.chdir('../code/')
import datetime 
import networkx as nx
from utils import *

timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 
newpath='../results/'

user_num=20
item_num=100
dimension=25
noise_list=[0.1, 0.25, 0.5]
loop_num=1
#user_f,item_f,pos,ori_signal,adj,lap=generate_random_graph(user_num, item_num, dimension)
user_f,item_f,pos,ori_signal,adj,lap=generate_GMRF(user_num, item_num, dimension)
error_dict_ols={}
error_dict_ridge={}
error_dict_graph_ridge={}
error_dict_gb={}
for loop in range(loop_num):
	for noise_scale in noise_list:
		print('noise scale=%s'%(noise_scale))
		noisy_signal=ori_signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)
		_lambda=1
		g_lambda=1
		iteration=3000

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




		for i in range(iteration):
			print('loop/total loop', loop, loop_num)
			print('iteration i=', i )
			print('noise scale=', noise_scale)
			if i<user_num:
				user=all_user[i]
				item=np.random.choice(all_item)
				user_dict[user].extend([item])
				item_sub_list=user_dict[user]
				user_list.extend([user])
				item_list.extend([item])
				user_list=list(np.unique(user_list))
				item_list=list(np.unique(item_list))
				user_nb=len(user_list)
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
				item_nb=len(item_list)
				sub_item_f=item_f[:, item_list]
				signal=noisy_signal[user_list][:,item_list]
				mask=np.ones((user_nb, item_nb))
				for ind, j in enumerate(user_list):
					remove_list=[x for x, xx in enumerate(item_list) if xx not in user_dict[j]]
					mask[ind, remove_list]=0
				signal=signal*mask
				sub_lap=lap[user_list][:,user_list]
				u_f=graph_ridge(user_nb, item_nb, dimension, sub_lap, sub_item_f, mask, signal, g_lambda)
				beta_graph_ridge[user_list]=u_f.copy()
				error_graph_ridge=np.linalg.norm(beta_graph_ridge-user_f)
				error_list_graph_ridge.extend([error_graph_ridge])
				# graph ridge iterative 
				beta_gb=graph_ridge_iterative_model(item_f, noisy_signal, lap, beta_gb, dimension, user_list, item_list, user_dict, g_lambda)
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