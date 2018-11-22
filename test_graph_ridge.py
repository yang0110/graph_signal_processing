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
#user_f,item_f,pos,ori_signal,adj,lap=generate_random_graph(user_num, item_num, dimension)
user_f,item_f,pos,ori_signal,adj,lap=generate_GMRF(user_num, item_num, dimension)


error_list_graph_ridge={noise:[] for noise in noise_list}
for noise_scale in noise_list:
	print('noise scale=%s'%(noise_scale))
	noisy_signal=ori_signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)
	g_lambda=0.2
	iteration=2000

	all_user=list(range(user_num))
	all_item=list(range(item_num))
	user_list=[]
	item_list=[]

	beta_graph_ridge=np.zeros((user_num, dimension))
	user_dict={a:[] for a in all_user}

	for i in range(iteration):
		print('iteration i=', i )
		print('noise_scale=', noise_scale)
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
			### Graph Ridge 
			user_list.extend([user])
			item_list.extend([item])
			user_list=list(np.unique(user_list))
			item_list=list(np.unique(item_list))
			user_nb=len(user_list)
			print('served user number', user_nb)
			item_nb=len(item_list)
			sub_item_f=item_f[:, item_list]
			signal=noisy_signal[user_list][:,item_list]
			mask=np.ones((user_nb, item_nb))
			for ind, j in enumerate(user_list):
				remove_list=[x for x, xx in enumerate(item_list) if xx not in user_dict[j]]
				mask[ind, remove_list]=0
			signal=signal*mask
			sub_lap=lap[user_list][:,user_list]
			print('sub_lap.size', sub_lap.shape)
			u_f=graph_ridge(user_nb, item_nb, dimension, sub_lap, sub_item_f, mask, signal, g_lambda)
			beta_graph_ridge[user_list]=u_f.copy()
			error_graph_ridge=np.linalg.norm(beta_graph_ridge-user_f)
			error_list_graph_ridge[noise_scale].extend([error_graph_ridge])



plt.figure()
for noise in noise_list:
	plt.plot(error_list_graph_ridge[noise], label='noise_scale=%s'%(noise))
plt.legend(loc=0)
plt.ylabel('MSE (Error)', fontsize=12)
plt.xlabel('#of sample (Size of training set)', fontsize=12)
plt.savefig(newpath+str(timeRun)+'Graph_Ridge_error_tune_noise_user_num_%s'%(user_num)+'.png', dpi=100)
plt.show()






