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

user_num=50
item_num=100
dimension=10
noise_list=[0.1, 0.25, 0.5]
# user_f,item_f,pos,ori_signal,adj,lap=generate_random_graph(user_num, item_num, dimension)
user_f,item_f,pos,ori_signal,adj,lap=generate_GMRF(user_num, item_num, dimension)
error_list_ols={noise:[] for noise in noise_list}
for noise_scale in noise_list:
	print('noise scale=%s'%(noise_scale))
	noisy_signal=ori_signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)
	_lambda=0.2
	iteration=1000

	all_user=list(range(user_num))
	all_item=list(range(item_num))
	user_list=[]
	item_list=[]

	beta_ols=np.zeros((user_num, dimension))
	user_dict={a:[] for a in all_user}

	for i in range(iteration):
		print('iteration i=', i )
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
			error_list_ols[noise_scale].extend([error_ols])


plt.figure()
for noise_scale in noise_list:
	plt.plot(error_list_ols[noise_scale], label='noise_scale=%s'%(noise_scale))
plt.legend(loc=0)
plt.ylabel('MSE (Error)', fontsize=12)
plt.xlabel('#of sample (Size of training set)', fontsize=12)
plt.savefig(newpath+str(timeRun)+'OLS_error_tune_noise_%s_user_num_%s_lambda_%s'%(noise_scale, user_num, _lambda)+'.png', dpi=100)
plt.clf()






