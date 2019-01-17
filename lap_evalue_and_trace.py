import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge
from utils import create_networkx_graph
from sklearn import datasets
path='../results/Bound/'

np.random.seed(2019)

user_num=50
dimension=10
item_num=2000
noise_level=0.1
d=2

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
Sigma=np.cov(item_f.T)
u,s,v=np.linalg.svd(Sigma)
sigma_min=np.min(s)
k=sigma_min/18

user_f=np.random.normal(size=(user_num, dimension))
user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
rank=np.linalg.matrix_rank(user_f)

ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
thrs_list=np.round(np.linspace((min_adj+max_adj)/2, max_adj, 5), decimals=4)
adj=ori_adj.copy()
thrs=0
adj[adj<=thrs]=0
lap=csgraph.laplacian(adj, normed=False)
lap_evalues, lap_vectors=np.linalg.eig(lap)
lap_evalues=np.sort(lap_evalues)
lap_min=np.min(lap_evalues)
lap_2=lap_evalues[1]
lap_user_fro=np.linalg.norm(np.dot(lap, user_f), 'fro')
lap_user_infty=np.linalg.norm(np.dot(lap, user_f), np.inf)
evalues_matrix=np.diag(lap_evalues)
lam_user_fro=np.linalg.norm(np.dot(evalues_matrix, user_f), 'fro')

def trace_norm(user_f, lap):
	norm=np.trace(np.dot(np.dot(user_f.T, lap), user_f))
	return norm 

cluster_std_list=np.arange(0.01, 20, 0.1)
norm_list=np.zeros(len(cluster_std_list))
norm_list2=np.zeros(len(cluster_std_list))
for index, cluster_std in enumerate(cluster_std_list):
	user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=cluster_std, shuffle=False, random_state=2019)
	user_f=Normalizer().fit_transform(user_f)
	adj=rbf_kernel(user_f)
	lap=csgraph.laplacian(adj, normed=False)
	lap_evalues, lap_evectors=np.linalg.eig(lap)
	lap_evalues=np.sort(lap_evalues)
	lam=np.diag(lap_evalues)
	norm_list[index]=trace_norm(user_f,lap)
	norm_list2[index]=lap_evalues[1]*np.linalg.norm(user_f, 'fro')**2

plt.plot(cluster_std_list, norm_list, label='trace')
plt.plot(cluster_std_list, norm_list2, label='evalues')
plt.legend(loc=0, fontsize=12)
plt.show()

user_f=np.random.normal(size=(user_num, dimension))
ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
thrs_list=np.round(np.linspace(min_adj, max_adj, 10),decimals=3)
trace_values=np.zeros(len(thrs_list))
eigen_norms_2=np.zeros(len(thrs_list))
eigen_norms_3=np.zeros(len(thrs_list))
eigen_norms_1=np.zeros(len(thrs_list))
for index, thrs in enumerate(thrs_list):
	adj=ori_adj.copy()
	adj[adj<=thrs]=0
	lap=csgraph.laplacian(adj, normed=False)
	lap_evalues, lap_evectors=np.linalg.eig(lap)
	lap_evalues=np.sort(lap_evalues)
	trace_values[index]=trace_norm(user_f, lap)
	eigen_norms_2[index]=lap_evalues[1]*np.linalg.norm(user_f, 'fro')**2
	eigen_norms_1[index]=lap_evalues[0]*np.linalg.norm(user_f, 'fro')**2
	eigen_norms_2[index]=lap_evalues[2]*np.linalg.norm(user_f, 'fro')**2

plt.plot(thrs_list, trace_values, label='trace')
plt.plot(thrs_list, eigen_norms_2, label='eigen_2_norms')
plt.legend(loc=0, fontsize=12)
plt.xlabel('threshold', fontsize=12)
plt.ylabel('norm', fontsize=12)
# plt.savefig(path+'trace_vs_lap_2_times_user_f'+'.png', dpi=200)
plt.show()

plt.plot(thrs_list, eigen_norms_1, label='eigen_1_norm')
plt.legend(loc=0, fontsize=12)
plt.xlabel('threshold', fontsize=12)
plt.ylabel('norm', fontsize=12)
# plt.savefig(path+'lap_1_times_user_f'+'.png', dpi=200)
plt.show()












