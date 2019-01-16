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

np.random.seed(2019)

def lambda_(noise_level, d, user_num, dimension, item_num):
	lam=8*np.sqrt(noise_level)*np.sqrt(d)*np.sqrt(user_num+dimension)/(user_num*item_num)
	return lam 

def ridge_bound_fro(lam, rank, I_user_fro, I_min, k):
	bound=lam*(np.sqrt(rank)+2*I_user_fro)/(k+lam*I_min)
	return bound 

def ridge_fixed_bound(lam, true_theta, Sigma, noise_level, item_num):



def ols_bound(item_num, noise_level, dimension):
	bound=noise_level*dimension/item_num
	return bound 


user_num=20
item_num=100
dimension=10
noise_leve=0.1
lam=0.1

user_f=np.random.normal(size=(user_num, dimension))
user_f=Normalizer().fit_transform(user_f)

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
Sigma=np.cov(item_f.T)
evalues, evectors=np.linalg.eig(Sigma)
clear_signal=np.dot(user_f, item_f.T)

x=item_f[0]
y=clear_signal[0,0]
theta=user_f[0]
theta_coeff=np.zeros(dimension)
a=np.zeros(dimension)
for i in range(dimension):
	theta_coeff[i]=np.dot(x, evectors[:,i])*y/evalues[i]
	a+=theta_coeff[i]*evectors[:,i]



ols_list=np.zeros(item_num)
for i in range(item_num):
	ols_list[i]=ols_bound(i+1, noise_level, dimension)

plt.plot(ols_list)
plt.show()




