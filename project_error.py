import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import scipy
import os 
os.chdir('Documents/code/')
import datetime 
import networkx as nx
from sklearn import datasets
from utils import *
path='../results/'

def each_user_error(R, S, lam, x, delta, evalue, I):
	V=lam*evalue*I
	V_t=np.dot(x.T, x)+V 
	a=np.sqrt(np.linalg.det(V_t))/np.sqrt(np.linalg.det(V))
	b=np.sqrt(2*np.log(a/delta))
	c=R*b+np.sqrt(lam)*S
	return c

def each_user_v_t(lam, x, I):
	v=np.dot(x.T, x)
	v_t=lam*I+v 
	return v_t 


user_num=50
dimension=10
item_num=1000
noise_level=0.1
lam=0.1
d=2

I=np.identity(dimension)

item_f=np.random.normal(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)


user_f=np.random.normal(size=(user_num, dimension))
user_f, _=datasets.make_blobs(n_samples=user_num, n_features=dimension, centers=5, cluster_std=0.1, shuffle=False, random_state=2019)
user_f=Normalizer().fit_transform(user_f)

clear_signal=np.dot(user_f, item_f.T)
noise=np.random.normal(size=(user_num, item_num), scale=noise_level)
noisy_signal=clear_signal+noise

x=item_f[:10]
vt=each_user_v_t(lam, x, I)

error_matrix=np.random.normal(size=(user_num, dimension))





