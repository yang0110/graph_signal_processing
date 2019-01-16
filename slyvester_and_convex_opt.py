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
import networkx as nx
#from utils import *
path='../results/'
np.random.seed(seed=2019)



user_num=20
item_num=100
dimension=10
noise_level=0.5
lam=0.1

user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)
ori_adj=rbf_kernel(user_f)
min_adj=np.min(ori_adj)
max_adj=np.max(ori_adj)
lap=csgraph.laplacian(ori_adj, normed=False)

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 

I=np.identity(user_num)

ridge_sly_array=np.zeros(item_num)
graph_sly_array=np.zeros(item_num)
graph_convex_array=np.zeros(item_num)
ridge_convex_array=np.zeros(item_num)
for i in range(item_num):
	print('i', i)
	x=item_f[:i+dimension,:]
	y=noisy_signal[:, :i+dimension]
	A=lam*lap 
	B=np.dot(x.T,x)
	C=np.dot(y,x)
	#graph ridge
	graph_sly_res=scipy.linalg.solve_sylvester(A,B,C)
	graph_sly_array[i]=np.linalg.norm(graph_sly_res-user_f, 'fro')
	graph_convex_res=graph_ridge_no_mask_convex(user_num, dimension, lap, x, y, lam)
	graph_convex_array[i]=np.linalg.norm(graph_convex_res-user_f, 'fro')
	#ridge
	AA=lam*I
	ridge_sly_res=scipy.linalg.solve_sylvester(AA,B,C)
	ridge_sly_array[i]=np.linalg.norm(ridge_sly_res-user_f, 'fro')
	ridge_convex_res=ridge_no_mask_convex(user_num, dimension, I, x, y, lam)
	ridge_convex_array[i]=np.linalg.norm(ridge_convex_res-user_f, 'fro')

plt.figure()
plt.plot(graph_sly_array, '+',label='graph-sly')
plt.plot(graph_convex_array, label='graph-convex')
plt.plot(ridge_sly_array, '+', label='ridge-sly')
plt.plot(ridge_convex_array, label='ridge-convex')
plt.legend(loc=0, fontsize=12)
plt.ylabel('Learning Error', fontsize=12)
plt.xlabel('Sample size')
plt.savefig(path+'slyvester_vs_cvxpy'+'.png', dpi=300)
plt.show()



