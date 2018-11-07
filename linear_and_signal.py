import numpy  as np 
import pandas as pd 
import cvxpy as cp
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csgraph
import os 
os.chdir('/Users/KGYaNG/Desktop/research/')

user_num=20
item_num=200
dimension=4
user_f=np.random.uniform(size=(user_num, dimension))
item_f=np.random.uniform(size=(item_num, dimension))
adj=rbf_kernel(user_f)
np.fill_diagonal(adj, 0)
lap=csgraph.laplacian(adj, normed=False)
signal=np.dot(item_f, user_f.T)
noise=np.random.normal(size=(item_num, user_num), scale=0.1)
noisy_signal=signal+noise


alpha_list=np.arange(0.01, 1, 0.1)
u=cp.Variable((user_num, dimension))
alpha=cp.Parameter(nonneg=True)
l_signal=cp.matmul(item_f, u.T)
reg=cp.sum([cp.quad_form(l_signal[i], lap) for i in range(item_num)])
loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
cons=[u<=1, u>=0]
error_list1=[]
signal_error_list1=[]
for a in alpha_list:
	print('a', a)
	alpha.value=a
	problem=cp.Problem(cp.Minimize(loss+alpha*reg), cons)
	problem.solve()
	sol=u.value
	error=np.linalg.norm(sol-user_f)
	error_list1.extend([error])
	sig=np.dot(item_f, sol.T)
	e=np.linalg.norm(sig-signal)
	signal_error_list1.extend([e])

u=cp.Variable((user_num, dimension))
l_signal=cp.matmul(item_f, u.T)
reg=cp.sum([cp.quad_form(u[:,i], lap) for i in range(dimension)])
loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
cons=[u<=1, u>=0]
error_list2=[]
signal_error_list2=[]
for a in alpha_list:
	print('a', a)
	alpha.value=a
	problem=cp.Problem(cp.Minimize(loss+alpha*reg), cons)
	problem.solve()
	sol=u.value
	error=np.linalg.norm(sol-user_f)
	error_list2.extend([error])
	sig=np.dot(item_f, sol.T)
	e=np.linalg.norm(sig-signal)
	signal_error_list2.extend([e])



def estimate_user_feature_cvx(user_num, item_num, dimension, item_f, lap, noisy_signal, alpha):
	u=cp.Variable((user_num, dimension))
	alp=cp.Parameter(nonneg=True)
	l_signal=cp.matmul(item_f, u.T)
	reg=cp.sum([cp.quad_form(u[:,i], lap) for i in range(dimension)])
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	cons=[u<=1, u>=0]
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg), cons)
	problem.solve()
	sol=u.value
	return sol 

solution=estimate_user_feature_cvx(user_num, item_num, dimension, item_f, lap, noisy_signal, 0.01)

y=cp.Variable((item_num, user_num))
alpha=cp.Parameter(nonneg=True)
loss=cp.pnorm(noisy_signal-y, p=2)**2
reg=cp.sum([cp.quad_form(y[i], lap) for i in range(item_num)])
error_list3=[]
signal_error_list3=[]
for a in alpha_list:
	print('a',a)
	alpha.value=a
	problem=cp.Problem(cp.Minimize(loss+alpha*reg), cons)
	problem.solve()
	sol=y.value
	e=np.linalg.norm(sol-signal)
	signal_error_list3.extend([e])
	a=np.dot(item_f.T, item_f)
	b=np.dot(item_f.T, sol)
	a_inv=np.linalg.inv(a)
	u=np.dot(a_inv, b)
	u_f=u.T
	error=np.linalg.norm(u_f-user_f)
	error_list3.extend([error])
















plt.figure(figsize=(5,5))
plt.plot(alpha_list, error_list3, label='ls', marker='*')
plt.plot(alpha_list, error_list1, label='tv')
plt.plot(alpha_list, error_list2, label='utv')
plt.legend(loc=1)
plt.ylabel('User Feature Learning Error')
plt.xlabel('alpha')
plt.show()

plt.figure(figsize=(5,5))
plt.plot(alpha_list, signal_error_list3, label='ls', marker='*')
plt.plot(alpha_list, signal_error_list1, label='tv')
plt.plot(alpha_list, signal_error_list2, label='utv')
plt.legend(loc=1)
plt.ylabel('Signal Learning Error')
plt.xlabel('alpha')
plt.show()



