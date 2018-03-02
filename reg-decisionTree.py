
import numpy as np

class RegressionDTree(object):

	def __init__(self, leaf_size = 1, verbose = False):
		self.leaf_size = leaf_size
		self.verbose = verbose
		pass

	def compute_best_feature(self, dataX, dataY):
		best_cor = -1
		best_col_idx = -1
		dataX_T = np.transpose(dataX)
		cor_array = np.array([])
		
		# here, the best X feature is the most correlated to Y.
		for col_idx in range(dataX.shape[1]):
			# ignore features with same values along the column : no correlation
			if not (self.contain_equal_values(dataX_T[col_idx])) :
				cor = np.absolute(np.corrcoef(dataX_T[col_idx], y = dataY)[0,1])
				cor_array = np.append(cor_array, cor)
				if (cor > best_cor) : 
					best_cor = cor
					best_col_idx = col_idx

		if self.verbose :
			print("---- cor : " + str(cor_array))
		
		return best_col_idx

	def compute_split_val(self, dataX, best_feature_idx): 
		
		median = np.median(dataX[:, best_feature_idx])
		maxi = dataX[:, best_feature_idx].max()
		mini = dataX[:, best_feature_idx].min()
		mean = np.mean(dataX[:, best_feature_idx])

		if (median == maxi) | (median == mini) :
			return mean
		
		return median

	def contain_equal_values(self, array):
		x0 = array[0]
		for x in array :
			if not np.array_equal(x0, x):
				return False

		return True

	def build_tree(self, dataX, dataY):

		# stoping condition 
		if ((dataX.shape[0] <= self.leaf_size) 
			| self.contain_equal_values(dataX)
			| self.contain_equal_values(dataY)):

			# return leaf with value = mean of y values
			return np.array([-1, np.mean(dataY), 0, 0])
		
		best_feature_idx = self.compute_best_feature(dataX, dataY)
		split_val = self.compute_split_val(dataX, best_feature_idx)

		if self.verbose : 
			print(" - New split on " + "X" + str(best_feature_idx) + " split_value = " + str(split_val))

		# build left and right branches
		left_mask = dataX[:, best_feature_idx] <= split_val
		right_mask = dataX[:, best_feature_idx] > split_val
		left_tree = self.build_tree(dataX[left_mask], dataY[left_mask])
		right_tree = self.build_tree(dataX[right_mask], dataY[right_mask])

		# add relative index
		if len(left_tree.shape) > 1 :
			righ_tree_rel_idx =  left_tree.shape[0] + 1
		else :
			righ_tree_rel_idx =  2
		root = np.array([best_feature_idx, split_val, 1, righ_tree_rel_idx])
		return np.vstack((root, left_tree, right_tree))

	def use_tree(self, row_idx, point) :
		# if in leaf : return leaf value
		node = self.tree[int(row_idx)]
		if (node[0] == -1) :
			return node[1]

		else :
			split_feature = int(node[0])
			split_value = node[1]
			left_tree_indx = row_idx + node[2]
			right_tree_indx = row_idx + node[3]
			if (point[split_feature] <= split_value) :
				return self.use_tree(left_tree_indx, point)
			else :
				return self.use_tree(right_tree_indx, point)
	

	def fit(self,dataX,dataY):
		"""
		@summary: Add training data to learner
		@param dataX: X values of data to add
		@param dataY: the Y training values
		"""
		self.tree = self.build_tree(dataX, dataY)
		if self.verbose :
			print(self.tree)
		
		
	def predict(self,points):
		"""
		@summary: Estimate a set of test points given the model we built.
		@param points: should be a numpy array with each row corresponding to a specific query.
		@returns the estimated values according to the saved model.
		"""
		Y = np.array([])
		for i in range(len(points)):
			point = points[i]
			output = self.use_tree(0, point)
			Y = np.append(Y, output)

		return Y

if __name__=="__main__":
	
