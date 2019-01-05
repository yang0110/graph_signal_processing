import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge
from utils import create_networkx_graph
from sklearn import datasets
path='../results/Bound/'


user_num=20
item_num=100
dimension=10

user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
thrs_list=np.round(np.linspace(min_adj, max_adj, 10), decimals=3)


clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num))
noisy_signal=clear_signal+noise 

def trace_norm(user_f, lap):
	trace=np.trace(np.dot(np.dot(user_f.T, lap), user_f))
	return trace

def lambda_lap_norm(user_fro, lambda_lap):
	norm=lambda_lap*user_fro
	return norm 

user_fro=np.linalg.norm(user_f, 'fro')**2
trace_array=np.zeros(len(thrs_list))
lambda_user_fro=np.zeros((dimension, len(thrs_list)))
for index, thrs in enumerate(thrs_list):
	adj=ori_adj.copy()
	adj[adj<=thrs]=0
	lap=csgraph.laplacian(adj, normed=False)
	lap_evalues, lap_evectors=np.linalg.eig(lap)
	lap_evalues=np.sort(lap_evalues)
	trace_array[index]=trace_norm(user_f, lap)
	for dim in range(dimension):
		lambda_lap=lap_evalues[dim]
		lambda_user_fro[dim, index]=lambda_lap_norm(user_fro, lambda_lap)

plt.figure()
plt.plot(trace_array,'k+', label='trace norm')
for dim in range(dimension):
	plt.plot(lambda_user_fro[dim+1, :], label=str(dim))

plt.legend(loc=0, fontsize=12)
plt.show()



