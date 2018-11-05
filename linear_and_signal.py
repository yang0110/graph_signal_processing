import numpy  as np 
import pandas as pd 
import cvxpy as cp
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse import csgraph
import os 
import datetime
os.chdir('/Users/KGYaNG/Desktop/research/')
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 
newpath='/Users/KGYaNG/Desktop/research/'+'linear_signaL_results/'+str(timeRun)+'/'
if not os.path.exists(newpath):
	os.makedirs(newpath)



user_num=20
item_num=200
dimension=10
alpha=0.01
user_f=np.random.uniform(size=(user_num, dimension))
item_f=np.random.uniform(size=(item_num, dimension))
adj=rbf_kernel(user_f)
np.fill_diagonal(adj, 0)
lap=csgraph.laplacian(adj, normed=False)
signal=np.dot(item_f, user_f.T)
noise=np.random.normal(size=(item_num, user_num), scale=0.1)
noisy_signal=signal+noise



def uatv_user_feature(user_num, item_num, dimension, lap, item_f, noisy_signal, alpha):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.matmul(item_f, u.T)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	reg=cp.sum([cp.quad_form(l_signal[i],lap) for i in range(item_num)])
	cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol 

def utv_user_feature(user_num, item_num, dimension, lap, item_f, noisy_signal, alpha):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.matmul(item_f, u.T)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	reg=cp.sum([cp.quad_form(u[:,i], lap) for i in range(dimension)])
	cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol




def stv_user_feature(user_num, item_num, dimension, lap, item_f, noisy_signal, alpha):
	y=cp.Variable((item_num, user_num))
	alp=cp.Parameter(nonneg=True)
	loss=cp.pnorm(noisy_signal-y, p=2)**2
	reg=cp.sum([cp.quad_form(y[i], lap) for i in range(item_num)])
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=y.value
	a=np.dot(item_f.T, item_f)
	a_inv=np.linalg.inv(a)
	b=np.dot(item_f.T, sol)
	u_f=np.dot(a_inv, b).T
	return sol, u_f 

def generate_signal(user_num, item_num, dimension, nois_scale):
	user_f=np.random.uniform(size=(user_num, dimension))
	item_f=np.random.uniform(size=(item_num, dimension))
	signal=np.dot(item_f, user_f.T)
	noise=np.random.normal(size=(item_num, user_num), scale=nois_scale)
	noisy_signal=signal+noise 
	adj=rbf_kernel(user_f)
	np.fill_diagonal(adj,0)
	lap=csgraph.laplacian(adj, normed=False)
	return user_f, item_f, signal, noisy_signal, adj, lap

noise_list=np.arange(0.1, 0.5, 0.1)
alpha=0.05
uatv_e_list=[]
utv_e_list=[]
stv_e_list=[]
for noise_scale in noise_list:
	print('noise', noise_scale)
	user_f, item_f, signal, noisy_signal, adj, lap=generate_signal(user_num, item_num, dimension, noise_scale)
	print('Date Generated!')
	uatv_user_f=uatv_user_feature(user_num, item_num, dimension, lap, item_f, noisy_signal, alpha)
	print('UATV')
	utv_user_f=utv_user_feature(user_num, item_num, dimension, lap, item_f, noisy_signal, alpha)
	print('UTV')
	stv_signal, stv_user_f=stv_user_feature(user_num, item_num, dimension, lap, item_f, noisy_signal, alpha)
	print('STV')

	uatv_e=np.mean(np.linalg.norm(uatv_user_f-user_f, axis=1))
	utv_e=np.mean(np.linalg.norm(utv_user_f-user_f, axis=1))
	stv_e=np.mean(np.linalg.norm(stv_user_f-user_f, axis=1))
	uatv_e_list.extend([uatv_e])
	utv_e_list.extend([utv_e])
	stv_e_list.extend([stv_e])

plt.figure(figsize=(5,5))
plt.plot(noise_list, stv_e_list, label='STV', marker='*')
plt.plot(noise_list, uatv_e_list, label='UATV')
plt.plot(noise_list, utv_e_list, label='UTV')
plt.legend(loc=1)
plt.ylabel('Learning Error')
plt.xlabel('Noise Level')
plt.tight_layout()
plt.savefig(newpath+'Noise_user_f_learning_error'+'.png')
plt.show()



noise_scale=0.1
alpha_list=np.arange(0.1, 1, 0.1)
uatv_e_list=[]
utv_e_list=[]
stv_e_list=[]
for alpha in alpha_list:
	print('alpha', alpha)
	user_f, item_f, signal, noisy_signal, adj, lap=generate_signal(user_num, item_num, dimension, noise_scale)
	print('Date Generated!')
	uatv_user_f=uatv_user_feature(user_num, item_num, dimension, lap, item_f, noisy_signal, alpha)
	print('UATV')
	utv_user_f=utv_user_feature(user_num, item_num, dimension, lap, item_f, noisy_signal, alpha)
	print('UTV')
	stv_signal, stv_user_f=stv_user_feature(user_num, item_num, dimension, lap, item_f, noisy_signal, alpha)
	print('STV')

	uatv_e=np.mean(np.linalg.norm(uatv_user_f-user_f, axis=1))
	utv_e=np.mean(np.linalg.norm(utv_user_f-user_f, axis=1))
	stv_e=np.mean(np.linalg.norm(stv_user_f-user_f, axis=1))
	uatv_e_list.extend([uatv_e])
	utv_e_list.extend([utv_e])
	stv_e_list.extend([stv_e])

plt.figure(figsize=(5,5))
plt.plot(alpha_list, stv_e_list, label='STV', marker='*')
plt.plot(alpha_list, uatv_e_list, label='UATV')
plt.plot(alpha_list, utv_e_list, label='UTV')
plt.legend(loc=1)
plt.ylabel('Learning Error')
plt.xlabel('Alpha')
plt.savefig(newpath+'Alpha_user_f_learning_error'+'.png')
plt.show()



