import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
import datetime 
import networkx as nx

user_num=100
item_num=1000
dimension=20 
iteration=2000
noise=0.1
item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
node_f=np.random.normal(size=(user_num, dimension))
adj=rbf_kernel(node_f)
thrs=0.5
adj[adj<=thrs]=0.0
lap=csgraph.laplacian(adj, normed=False)
cov=np.linalg.pinv(lap)
user_f=np.random.multivariate_normal(np.zeros(user_num), cov, size=dimension)
user_f=Normalizer().fit_transform(user_f)

pool_size=25
user_array=np.random.choice(np.arange(user_num), size=iteration)
item_array=np.random.choice(np.arange(item_num), size=iteration*pool_size).reshape((iteration, pool_size))

clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 
mask=np.zeros((user_num, item_num))

es_user_fs=np.zeros((user_num, dimension))
user_covs={}
served_users=[]
for i in range(iteration):
	user=user_array[i]
	if user is in served_users:
		pass 
	else:
		user_covs[user]=np.zeros((dimension, dimension))
		served_users.extend([user])

	item_pool=item_array[i]
	es_user_f=es_user_fs[i]



