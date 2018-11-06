import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx 
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from scipy.sparse import csgraph

def ols(x, y):
	cov=np.dot(x.T, x)
	beta_ols=np.dot(np.linalg.inv(cov), np.dot(x.T,y)).T
	return beta_ols

def ridge(x,y, _lambda, dimension):
	cov=np.dot(x.T,x)
	temp1=cov+_lambda*np.identity(dimension)
	temp2=np.linalg.inv(temp1)
	beta_ridge=np.dot(temp2, np.dot(x.T,y)).T
	return beta_ridge

def graph_ridge(x, y, _lambda, dimension, lap, user_num):
	cov=np.linalg.norm(np.dot(x.T, x))*np.identity(user_num)
	temp1=cov+_lambda*lap
	temp2=np.linalg.inv(temp1)
	beta_graph=np.dot(temp2, np.dot(x.T,y.T)).T
	return beta_graph

def mab_bound(x, _lambda, lap, sigma, delta, user_num):
	cov=np.linalg.norm(np.dot(x, x.T))*np.identity(user_num)
	v=cov+_lambda*lap
	temp1=np.sqrt(np.linalg.det(v))
	temp2=delta*np.sqrt(np.linalg.det(_lambda*lap))
	temp3=np.sqrt(2*np.log(temp1/temp2))
	temp4=np.sqrt(np.linalg.det(sigma*lap))
	bound=temp3+temp4
	# print('temp1', temp1)
	# print('temp2', temp2)
	# print('temp3', temp3)
	# print('temp4', temp4)
	# print('bound', bound)
	return bound




user_num=20
item_num=200
dimension=10
sigma=0.25
_lambda=0.5
delta=0.05
iteration=2000
user_f=np.random.uniform(size=(user_num, dimension))
item_f=np.random.uniform(size=(item_num, dimension))
item_f=Normalizer().fit_transform(item_f)
true_adj=rbf_kernel(user_f)
np.fill_diagonal(true_adj, 0)
lap=csgraph.laplacian(true_adj, normed=False)
true_signal=np.dot(item_f, user_f.T)
noisy_signal=true_signal+np.random.normal(size=(item_num, user_num), scale=sigma)
user_seq=np.random.choice(list(range(user_num)), size=iteration, replace=True)
item_seq=np.random.choice(list(range(item_num)), size=iteration, replace=True)
user_beta=np.zeros((user_num, dimension))
beta_ols=np.zeros((user_num, dimension))
beta_ridge=np.zeros((user_num, dimension))
user_dict={i:[] for i in range(user_num)}
received_signals=np.zeros((user_num, item_num))
error_list=[]
error_list_ols=[]
error_list_ridge=[]
bound_list=[]
for i in range(iteration):
	print('i', i)
	user=user_seq[i]
	item=item_seq[i]
	user_list=np.unique(user_seq[:i+1])
	sub_lap=lap[user_list,:][:,user_list]
	user_dict[user].extend([item])
	received_signals[user, item]=noisy_signal.T[user, item]
	y=received_signals[user_list]
	x=item_f.copy().T
	for j in range(dimension):
		a=np.dot(x[j,:].T, x[j,:])*np.identity(len(user_list))+_lambda*sub_lap
		a_inv=np.linalg.inv(a)
		_sum=np.zeros((len(user_list), item_num))
		for jj in range(dimension):
			if jj==j:
				pass 
			else:
				_sum+=np.dot(user_beta[user_list,jj].reshape((len(user_list), 1)), x[jj,:].reshape((1, item_num)))
		e=y-_sum
		b=np.dot(e,x[j,:].T)
		user_beta[user_list,j]=np.dot(a_inv,b)

	error=np.linalg.norm(user_f-user_beta)
	error_list.extend([error])


	##OLS
	x=item_f[user_dict[user]]
	y=received_signals[user, user_dict[user]]
	beta_ols[user]=ols(x,y)
	error_ols=np.linalg.norm(beta_ols-user_f)
	error_list_ols.extend([error_ols])

	beta_ridge[user]=ridge(x,y,_lambda, dimension)
	error_ridge=np.linalg.norm(beta_ridge-user_f)
	error_list_ridge.extend([error_ridge])
	bound=mab_bound(x, _lambda, lap, sigma, delta, user_num)
	bound_list.extend([bound])

plt.figure()
plt.plot(error_list[500:], label='graph-ridge')
plt.plot(error_list_ols[500:], label='ols')
plt.plot(error_list_ridge[500:], label='ridge')
plt.plot(bound_list[500:], label='bound')
plt.legend(loc=0)
plt.show()



