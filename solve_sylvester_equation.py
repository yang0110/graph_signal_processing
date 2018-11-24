import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import os 
os.chdir('C:/Kaige_Research/Code/GSP/code/')
import datetime 
import networkx as nx
from utils import *

timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 
newpath='../results/'

user_num=50
item_num=100
dimension=10
noise_list=[0.1, 0.25, 0.5]
noise_scale=0.1
user_f,item_f,pos,ori_signal,adj,lap=generate_GMRF(user_num, item_num, dimension)
noisy_signal=ori_signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)

Y=noisy_signal.copy()
X=item_f.copy()
U_true=user_f.copy()
U_norm=np.linalg.norm(U_true)
A=noise_scale*lap 
A_norm=np.linalg.norm(A)
B=-(np.dot(X, X.T))
B_norm=np.linalg.norm(B)
C=np.dot(Y, X.T)
U=np.zeros((user_num, dimension))
k_max=1000
error_list=[]
for k in range(k_max):
	print('k',k)
	Z=C+np.dot(U, B)
	U=np.linalg.solve(A, Z)
	error=np.linalg.norm(U_true, U)
	error_list.extend([error])
	bound=(A_norm*B_norm)**K*U_norm
	bound_list.extend([bound])
	stop=np.linalg.norm(np.dot(A, U)-Z)
	if stop<10**(-2):
		print('Break, |AU-Z|=', stop)
		break 
	else:
		pass 

plt.figure()
plt.plot(bound_list, label='Error Bound')
plt.plot(error_list, label='Erroe')
plt.legend(loc=0, fontsize=12)
plt.show()

