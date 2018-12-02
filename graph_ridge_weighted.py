Graph ridge weighted
weights[user]=1/np.trace(noise_scale*np.linalg.inv(np.dot(X.T,X)+0.01*np.identity(dimension)))
sub_weights=np.diag(weights[user_list])
gb_weight=graph_ridge_weighted(user_nb, item_nb, dimension, sub_weights, sub_lap, sub_item_f, mask, signal, g_lambda)
beta_graph_ridge_weighted[user_list]=gb_weight.copy()
error_graph_ridge_weighted=np.linalg.norm(beta_graph_ridge_weighted-user_f)
error_list_graph_ridge_weighted.extend([error_graph_ridge_weighted])