import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
# os.chdir('Documents/research/code/')
import datetime 
import networkx as nx
from bandit_models import LinUCB, Graph_ridge
from utils import create_networkx_graph
from sklearn import datasets
path='../results/Bound/'

def each_user_error(R, S, lam, x, delta, evalue, I):
	V=lam*evalue*I
	V_t=np.dot(x.T, x)+V 
	a=np.sqrt(np.linalg.det(V_t))/np.sqrt(np.linalg.det(V))
	b=np.sqrt(2*np.log(a/delta))
	c=R*b+np.sqrt(lam)*S
	return c 


np.random.seed(2019)

user_num=50
dimension=10
item_num=200
lam=0.01
noise_level=0.5
S=1
R=noise_level
delta=0.05
I=np.identity(dimension)

user_f=np.random.normal(size=(user_num, dimension))
user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)
adj=rbf_kernel(user_f)
lap=csgraph.laplacian(adj, normed=False)
lap_evalues, lap_evectors=np.linalg.eig(lap)
lap_evalues=np.sort(lap_evalues)
evalues_matrix=np.diag(lap_evalues)
Lambda=evalues_matrix.copy()

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)

clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise 


graph_ridge_simple_emp_error=np.zeros(item_num)
graph_ridge_emp_error=np.zeros(item_num)
ridge_emp_error=np.zeros(item_num)
bounds=np.zeros((user_num, item_num))
for i in range(item_num):
	x=item_f[:i+1, :]
	y=noisy_signal[:, :i+1]
	A=lam*Lambda
	A2=lam*lap
	AA=lam*np.identity(user_num)
	B=np.dot(x.T, x)
	C=np.dot(y, x)
	graph_simple_ridge=scipy.linalg.solve_sylvester(A,B,C)
	graph_ridge=scipy.linalg.solve_sylvester(A2,B,C)
	ridge=scipy.linalg.solve_sylvester(AA, B, C)
	ridge_emp_error[i]=np.linalg.norm(ridge-user_f, 'fro')
	graph_ridge_simple_emp_error[i]=np.linalg.norm(graph_simple_ridge-user_f, 'fro')
	graph_ridge_emp_error[i]=np.linalg.norm(graph_ridge-user_f, 'fro')
	for j in range(user_num):
		bounds[j, i]=each_user_error(R, S, lam, x, delta, 1, I)

the_error=np.sum(bounds, axis=0)

plt.figure()
plt.plot(ridge_emp_error, label='ridge')
plt.plot(graph_ridge_emp_error, label='graph ridge')
plt.plot(graph_ridge_simple_emp_error, label='graph simple ridge')
plt.plot(the_error, label='Theoritical error')
plt.legend(loc=0, fontsize=12)
plt.show()














