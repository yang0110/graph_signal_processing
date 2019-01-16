import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/research/code/')
import datetime 
from sklearn import datasets
import networkx as nx
from utils import *
path='../results/Bound/'

user_num=100
item_num=1000
dimension=25
noise_level=0.1
cluster_std=20

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)


thrs_num=100
std_list=np.linspace(0.1, 1, 5)
ratio_array=np.zeros((len(std_list), thrs_num))
lap_infty_array=np.zeros((len(std_list), thrs_num))
lap_trace_array=np.zeros((len(std_list),  thrs_num))
edge_num_array=np.zeros((len(std_list), thrs_num))
for i, cluster_std in enumerate(std_list):
	user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=cluster_std, shuffle=False, random_state=2019)
	user_f=Normalizer().fit_transform(user_f)
	adj=rbf_kernel(user_f)
	min_ad=np.min(adj)
	max_adj=np.max(adj)
	thrs_list=np.round(np.linspace(min_adj, max_adj, thrs_num), decimals=4)
	for j, thrs in enumerate(thrs_list):
		new_adj=adj.copy()
		new_adj[new_adj<=thrs]=0
		graph, edge_num=create_networkx_graph(user_num, new_adj)
		lap=csgraph.laplacian(new_adj, normed=False)
		lap_infty=np.linalg.norm(lap, np.inf)
		lap_trace=np.trace(lap)
		ratio_array[i, j]=lap_infty/lap_trace
		lap_infty_array[i,j]=lap_infty
		lap_trace_array[i,j]=lap_trace
		edge_num_array[i,j]=edge_num

for ind, std in enumerate(std_list):
	plt.figure()
	plt.plot(edge_num_array[ind], ratio_array[ind], label=str(std))
	plt.legend(loc=0, fontsize=12)
	plt.show()







