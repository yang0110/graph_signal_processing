import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import Normalizer
from scipy.sparse import csgraph 
import os 
import datetime 
def ols(x,y):
	cov=np.dot(x.T,x)
	temp2=np.linalg.inv(cov)
	beta=np.dot(temp2, np.dot(x.T,y)).T
	return beta 

def ridge(x,y, _lambda, dimension):
	cov=np.dot(x.T,x)
	temp1=cov+_lambda*np.identity(dimension)
	temp2=np.linalg.inv(temp1)
	beta=np.dot(temp2, np.dot(x.T,y)).T
	return beta 

def graph_ridge(user_num, item_num, dimension, lap, item_f, mask, noisy_signal, alpha):
	u=cp.Variable((user_num, dimension))
	l_signal=cp.multiply(cp.matmul(u, item_f),mask)
	loss=cp.pnorm(noisy_signal-l_signal, p=2)**2
	reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(dimension)])
	cons=[u<=1, u>=0]
	alp=cp.Parameter(nonneg=True)
	alp.value=alpha 
	problem=cp.Problem(cp.Minimize(loss+alp*reg))
	problem.solve()
	sol=u.value 
	return sol 

def generate_data(user_num, item_num, dimension, noise_scale):
	user_f=np.random.normal(size=(user_num, dimension))# N*K
	user_f=Normalizer().fit_transform(user_f)
	item_f=np.random.normal(size=(dimension, item_num))# K*M
	item_f=Normalizer().fit_transform(item_f.T).T
	signal=np.dot(user_f, item_f)# N*M 
	noisy_signal=signal+np.random.normal(size=(user_num, item_num), scale=noise_scale)
	true_adj=rbf_kernel(user_f)
	np.fill_diagonal(true_adj, 0)
	lap=csgraph.laplacian(true_adj, normed=False)
	return user_f, item_f, noisy_signal, true_adj, lap 


os.chdir('Users/KGYaNG/Desktop/research/')
timeRun = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S') 
newpath='/Users/KGYaNG/Desktop/research/'+'linear_signaL_results/'+str(timeRun)+'/'
if not os.path.exists(newpath):
	os.makedirs(newpath)


user_num=50
item_num=500
dimension=10
noise_scale=0.25
user_f, item_f, noisy_signal, adj, lap=generate_data(user_num, item_num, dimension, noise_scale)

_lambda=0.5
iteration=500

all_user=list(range(user_num))
all_item=list(range(item_num))
user_list=[]
item_list=[]

beta_ols=np.zeros((user_num, dimension))
beta_ridge=np.zeros((user_num, dimension))
beta_graph_ridge=np.zeros((user_num, dimension))
user_dict={a:[] for a in all_user}
error_list_ols=[]
error_list_ridge=[]
error_list_graph_ridge=[]



for i in range(iteration):
	print('iteration i= ', i )
	user=np.random.choice(all_user)
	item=np.random.choice(all_item)
	user_dict[user].extend([item])
	item_sub_list=user_dict[user]
	X=item_f[:, item_sub_list].T
	y=noisy_signal[user, item_sub_list]
	### OLS
	beta_ols[user]=ridge(X, y, 0.01, dimension)
	error_ols=np.linalg.norm(beta_ols-user_f)
	error_list_ols.extend([error_ols])
	### Ridge 
	beta_ridge[user]=ridge(X, y, _lambda, dimension)
	error_ridge=np.linalg.norm(beta_ridge-user_f)
	error_list_ridge.extend([error_ridge])
	### Graph Ridge 
	user_list.extend([user])
	item_list.extend([item])
	user_list=list(np.unique(user_list))
	item_list=list(np.unique(item_list))
	user_nb=len(user_list)
	item_nb=len(item_list)
	item_feature=item_f[:, item_list]
	signal=noisy_signal[user_list][:,item_list]
	mask=np.ones((user_nb, item_nb))
	for ind, j in enumerate(user_list):
		remove_list=[x for x, xx in enumerate(item_list) if xx not in user_dict[j]]
		mask[ind, remove_list]=0
	signal=signal*mask
	sub_lap=lap[user_list][:,user_list]
	u_f=graph_ridge(user_nb, item_nb, dimension, sub_lap, item_feature, mask, signal, _lambda)
	beta_graph_ridge[user_list]=u_f.copy()
	error_graph_ridge=np.linalg.norm(beta_graph_ridge-user_f)
	error_list_graph_ridge.extend([error_graph_ridge])




plt.figure()
plt.plot(error_list_ols, label='OLS')
plt.plot(error_list_ridge, label='Ridge')
plt.plot(error_list_graph_ridge, label='Graph-ridge')
plt.legend(loc=0, fontsize=12)
plt.ylabel('MSE')
plt.xlabel('#of sample')
plt.show()















