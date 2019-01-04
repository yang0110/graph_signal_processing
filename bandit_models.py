import numpy as np 
import cvxpy as cp 
import matplotlib.pyplot as plt
from scipy.sparse import csgraph 
import scipy
import os 
from sklearn.metrics.pairwise import rbf_kernel
import cvxpy as cp

class LinUCB():
	def __init__(self, user_num, item_num, dimension, pool_size, user_features, item_features,noisy_signal, alpha, confidence_inter):
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.item_features=item_features
		self.user_features=user_features
		self.es_user_f=np.zeros((user_num, dimension))
		self.noisy_signal=noisy_signal
		self.user_cov={}
		self.bias={}
		self.served_users=[]
		self.alpha=alpha
		self.c_t=confidence_inter

	def update_user_features(self,  selected_user, picked_item, reward):
		self.user_cov[selected_user]+=np.outer(self.item_features[picked_item], self.item_features[picked_item])+self.alpha*np.identity(self.dimension)
		self.bias[selected_user]+=self.item_features[picked_item]*reward
		self.es_user_f[selected_user]=np.dot(np.linalg.pinv(self.user_cov[selected_user]), self.bias[selected_user])


	def choose_item(self, selected_user, item_pool):
		a=self.item_features[item_pool]
		mean=np.dot(a,self.es_user_f[selected_user])
		b=np.linalg.pinv(self.user_cov[selected_user])
		weighted_norm=np.diagonal(np.sqrt(np.dot(a, np.dot(b, a.T))))
		sum_=mean+self.c_t*weighted_norm
		index=np.argmax(sum_)
		picked_item=item_pool[index]
		reward=self.noisy_signal[selected_user, picked_item]
		best_reward=np.max(self.noisy_signal[selected_user, item_pool])
		regret=best_reward-reward
		return picked_item, reward, regret

	def run(self, random_user_list, random_item_pool, iteration):
		cum_regret=[0]
		error_list=[]
		error_all_array=np.zeros((iteration, self.user_num))
		for i in range(iteration):
			print('iteration', i)
			selected_user=random_user_list[i]
			item_pool=random_item_pool[i]
			if selected_user in self.served_users:
				pass 
			else:
				self.served_users.extend([selected_user])
				self.user_cov[selected_user]=np.zeros((self.dimension, self.dimension))
				self.bias[selected_user]=np.zeros(self.dimension)
			picked_item, reward, regret=self.choose_item(selected_user, item_pool)
			self.update_user_features(selected_user, picked_item, reward)
			cum_regret.extend([cum_regret[-1]+regret])
			error=np.linalg.norm(self.es_user_f-self.user_features, 'fro')
			error_list.extend([error])
			error_all_user=np.linalg.norm(self.es_user_f-self.user_features, axis=1)
			error_all_array[i,:]=error_all_user
		return cum_regret, error_list, error_all_array


class Graph_ridge():
	def __init__(self, user_num, item_num, dimension, lap, adj, pool_size, user_features, item_features, noisy_signal, alpha, confidence_inter):
		self.lap=lap+np.identity(user_num)
		self.adj=adj
		self.mask=np.zeros((user_num, item_num))
		self.dimension=dimension
		self.user_num=user_num
		self.item_num=item_num
		self.item_features=item_features
		self.user_features=user_features
		self.es_user_f=np.zeros((user_num, dimension))
		self.noisy_signal=noisy_signal
		self.user_cov={}
		self.bias={}
		self.served_users=[]
		self.served_items=[]
		self.alpha=alpha
		self.c_t=confidence_inter
		self.user_mask=np.zeros((user_num, dimension))
		self.item_mask=np.zeros((item_num, dimension))

	def update_user_features(self, selected_user, picked_item, reward):
		self.user_cov[selected_user]+=np.outer(self.item_features[picked_item], self.item_features[picked_item])+self.alpha*np.identity(self.dimension)
		self.bias[selected_user]+=self.item_features[picked_item]*reward
		#### Update all users in each iteration
		# u=cp.Variable((self.user_num, self.dimension))
		# l_signal=cp.multiply(cp.matmul(u, self.item_features.T), self.mask)
		# loss=cp.norm(cp.multiply(self.noisy_signal, self.mask)-l_signal, 'fro')**2
		# reg=cp.sum([cp.quad_form(u[:,d], self.lap) for d in range(self.dimension)])
		# alp=cp.Parameter(nonneg=True)
		# alp.value=self.alpha 
		# problem=cp.Problem(cp.Minimize(loss+alp*reg))
		# problem.solve()
		# self.es_user_f=u.value 
		##### Only update neighbors at each iteration
		neighbors=np.where(self.adj[selected_user]>0)[0]
		user_index=np.where(neighbors==selected_user)[0]
		print('user_index', user_index)
		neighbor_num=len(neighbors)
		print('neighbor_num', neighbor_num)
		lap=self.lap[neighbors][:, neighbors]
		item_f=self.item_features[self.served_items]
		mask=self.mask[neighbors][:,self.served_items]
		noisy_signal=self.noisy_signal[neighbors][:, self.served_items]
		self.es_user_f[neighbors]=self.graph_ridge_solver(neighbor_num, mask, item_f, noisy_signal, lap)

	def graph_ridge_solver(self, neighbor_num, mask, item_f, noisy_signal, lap):
		u=cp.Variable((neighbor_num, self.dimension))
		l_signal=cp.multiply(cp.matmul(u, item_f.T), mask)
		loss=cp.norm(cp.multiply(noisy_signal, mask)-l_signal, 'fro')**2
		reg=cp.sum([cp.quad_form(u[:,d], lap) for d in range(self.dimension)])
		alp=cp.Parameter(nonneg=True)
		alp.value=self.alpha 
		problem=cp.Problem(cp.Minimize(loss+alp*reg))
		problem.solve()	
		solution=u.value
		return solution


	def choose_item(self, selected_user, item_pool):
		a=self.item_features[item_pool]
		mean=np.dot(a,self.es_user_f[selected_user])
		b=np.linalg.pinv(self.user_cov[selected_user])
		weighted_norm=np.diagonal(np.sqrt(np.dot(a, np.dot(b, a.T))))
		sum_=mean+self.c_t*weighted_norm
		index=np.argmax(sum_)
		picked_item=item_pool[index]
		reward=self.noisy_signal[selected_user, picked_item]
		best_reward=np.max(self.noisy_signal[selected_user, item_pool])
		regret=best_reward-reward
		return picked_item, reward, regret

	def run(self, random_user_list, random_item_pool, iteration):
		cum_regret=[0]
		error_list=[]
		error_all_array=np.zeros((iteration, self.user_num))
		trace_list=[]
		for i in range(iteration):
			print('iteration', i)
			selected_user=random_user_list[i]
			item_pool=random_item_pool[i]
			if selected_user in self.served_users:
				pass 
			else:
				self.served_users.extend([selected_user])
				self.user_cov[selected_user]=np.zeros((self.dimension, self.dimension))
				self.bias[selected_user]=np.zeros(self.dimension)
			picked_item, reward, regret=self.choose_item(selected_user, item_pool)
			if picked_item in self.served_items:
				pass 
			else:
				self.served_items.extend([picked_item])
			self.mask[selected_user, picked_item]=1
			self.user_mask[selected_user,:]=1
			self.item_mask[picked_item,:]=1
			self.update_user_features(selected_user, picked_item, reward)
			cum_regret.extend([cum_regret[-1]+regret])
			error=np.linalg.norm(self.es_user_f-self.user_features, 'fro')
			error_list.extend([error])
			error_all_user=np.linalg.norm(self.es_user_f-self.user_features, axis=1)
			error_all_array[i,:]=error_all_user
			trace=np.trace(np.dot(np.dot((self.es_user_f-self.user_features).T, self.lap), self.user_features))
			trace_list.extend([trace])

		return cum_regret, error_list, error_all_array, trace_list








