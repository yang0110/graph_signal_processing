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
path='../results/Bound/theoretical bound/'

np.random.seed(2019)

user_num=50 
dimension=10
true_cluster_std=1
user_f=np.random.normal(size=(user_num, dimension))
true_user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=true_cluster_std, shuffle=False, random_state=2019)
true_user_f=Normalizer().fit_transform(true_user_f)

cluster_std_list=np.arange(0.1,5,0.1)
fro_list=np.zeros(len(cluster_std_list))
for index, cluster_std in enumerate(cluster_std_list):
	user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5,center_box=(-5,5), cluster_std=cluster_std, shuffle=False, random_state=2019)
	user_f=Normalizer().fit_transform(user_f)
	adj=rbf_kernel(user_f)
	lap=csgraph.laplacian(adj, normed=False)
	lap_user_fro=np.linalg.norm(np.dot(lap, true_user_f), 'fro')**2
	fro_list[index]=lap_user_fro 
	if cluster_std in [0.1,1.0,2.0,4.0]:
		print('cluster std', cluster_std)
		plt.figure(figsize=(5,5))
		plt.pcolor(adj)
		plt.colorbar()
		plt.savefig(path+'lap_user_fro_cluster_std_%s'%(cluster_std)+'.png', dpi=100)
		plt.show()
	else:
		pass

plt.figure(figsize=(5,5))
plt.plot(cluster_std_list, fro_list)
plt.xlabel('Cluster standard deviation', fontsize=12)
plt.ylabel('Laplacian user frobenious', fontsize=12)
plt.tight_layout()
plt.savefig(path+'lap_user_fro_vs_cluster_std'+'.png', dpi=200)
plt.show()

