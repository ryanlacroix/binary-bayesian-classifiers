import csv
import random
import copy
import estimate_dep_tree as dpnt
from statistics import mean

# Creates shuffled set of every class
def create_dataset(file_names):
	dataset_full = []
	# Keep track of which class is being dealt with
	class_num = 1
	for filename in file_names:
		class_name = 'class' + str(class_num)
		with open(filename, 'r') as f:
			# Convert every field in CSV file to integers
			dat = [list(map(int,rec)) for rec in csv.reader(f, delimiter=',')]
			# Add the class to the end of each
			for line in dat:
				dataset_full.append(line)
		class_num += 1

	random.shuffle(dataset_full)
	with open('dataset_full_shuffled.csv', 'w', newline='') as f:
		output = csv.writer(f)
		output.writerows(dataset_full)
	return dataset_full

def get_datasets(file_names):
	datasets = []
	for filename in file_names:
		with open(filename, 'r') as f:
			dat = [list(map(int,rec)) for rec in csv.reader(f, delimiter=',')]
			datasets.append(dat)
	return datasets

# Perform k-fold cross validation to train and test the datasets
def k_fold_validation(full_set, datasets, fold_num, cl_type, show_trees = False):
	confusion_matrix = [[0 for x in range(len(datasets))] for y in range(len(datasets))] 
	# Store each possible combination of class index
	class_combos = []
	for i in range(0, len(datasets)):
		for j in range(0, len(datasets)):
			class_combos.append((i,j))

	# Construct dependence trees if needed
	if cl_type == "dependent":
		dep_trees = []
		for class_set in datasets:
			tree = dpnt.estimate_dep_tree(class_set, len(class_set[0]), show_trees)
			dep_trees.append(tree)

	# Depending on the dataset, it is sometimes beneficial to 
	# instead estimate one tree based on the entire dataset.
	# Uncomment the code below to instead use this technique.

	# Create four identical tree structures based on total set
	# dep_trees = []
	# for i in datasets:
	# 	tree = dpnt.estimate_dep_tree(full_set, len(datasets[0][0]), False)
	# 	dep_trees.append(tree)

	# Compare each possible pair of classes
	for class_pair in class_combos:
		# Calculate size of folds and samples
		total_samples1 = len(datasets[class_pair[0]])
		total_samples2 = len(datasets[class_pair[1]])
		fold_size1 = int(total_samples1 / fold_num)
		fold_size2 = int(total_samples2 / fold_num)
		correct_ratio_list1 = []
		correct_ratio_list2 = []

		for k in range(0, fold_num):
			class1 = datasets[class_pair[0]]
			class2 = datasets[class_pair[1]]
			# Data currently used for testing purposes
			test_range1 = (k*fold_size1,(k*fold_size1)+fold_size1)
			test_range2 = (k*fold_size2,(k*fold_size2)+fold_size2)
			
			# Seperate training and testing sets
			test_set1 = class1[test_range1[0]:test_range1[1]]
			test_set2 = class2[test_range2[0]:test_range2[1]]
			
			training_set1 = class1[:test_range1[0]] + class1[test_range1[1]:]
			training_set2 = class2[:test_range2[0]] + class2[test_range2[1]:]

			# Calculate probability of feature occurring in either set
			class1_probs = get_feature_probs(training_set1)
			class2_probs = get_feature_probs(training_set2)

			if cl_type == "dependent":
				dpnt.build_tree_probs(dep_trees[class_pair[0]], training_set1)
				dpnt.build_tree_probs(dep_trees[class_pair[1]], training_set2)

			correct_incorrect_ratio1 = [0,0]
			correct_incorrect_ratio2 = [0,0]

			# Test each set
			for sample in test_set1:
				if cl_type == "independent":
					guess = guess_class_from_sample(class1_probs, class2_probs, sample)
				elif cl_type == "dependent":
					guess = dpnt_guess_class_from_sample(dep_trees[class_pair[0]], dep_trees[class_pair[1]], sample)
				# Add the result into the confusion matrix
				if guess == 1:
					confusion_matrix[class_pair[0]][class_pair[0]] += 1
				else:
					confusion_matrix[class_pair[0]][class_pair[1]] += 1

			for sample in test_set2:
				if cl_type == "independent":
					guess = guess_class_from_sample(class1_probs, class2_probs, sample)
				elif cl_type == "dependent":
					guess = dpnt_guess_class_from_sample(dep_trees[class_pair[0]], dep_trees[class_pair[1]], sample)
				if guess == 2:
					confusion_matrix[class_pair[1]][class_pair[1]] += 1
				else:
					confusion_matrix[class_pair[1]][class_pair[0]] += 1
	
	# Prepare the matrix for display and output performance
	normalize_matrix(confusion_matrix)
	printMatrix(confusion_matrix)
	write_matrix_csv(confusion_matrix, cl_type)
	print("Average accuracy: ",avg_accuracy(confusion_matrix))

# Calculate average accuracy of the classifier based on confusion matrix
def avg_accuracy(matrix):
	totals = []
	for i in range(0, len(matrix[0])):
		for j in range(0, len(matrix[0])):
			if i == j:
				totals.append(matrix[i][j]) 
	return mean(totals)

# Display the matrix in the console
def printMatrix(matrix):
	for i in range(0, len(matrix[0])):
		for j in range(0, len(matrix[0])):
			print(str(matrix[j][i])+" ", end='')
		print('')

# Write the matrix to a .csv file
# cl_type denotes classifier type used
def write_matrix_csv(matrix, cl_type):
	mat = copy.deepcopy(matrix)
	columns = []
	for i in range(1,len(matrix[0])+1):
		columns.append('Class ' + str(i))
	with open(cl_type + '_bayes_conf.csv', 'w', newline='') as f:
		output = csv.writer(f)
		output.writerow(columns)
		for i in range(0, len(matrix[0])):
			mat[i].append(columns[i])
			output.writerow(mat[i])

# Normalize the matrix such that each column
# represents a percentage within each row.
def normalize_matrix(matrix):
	line_totals = []
	# Get totals from each line
	for i in range(0, len(matrix[0])):
		new_total = 0
		for j in range(0, len(matrix[0])):
			new_total += matrix[j][i]
		line_totals.append(new_total)
	# Reset values to percentages of line total
	for i in range(0, len(matrix[0])):
		for j in range(0, len(matrix[0])):
			matrix[j][i] = int((matrix[j][i]/max(line_totals[i], 0.001))*100)
		
# Guess class using independent Bayesian classification
def guess_class_from_sample(feat_probs1, feat_probs2, sample):
	# Determine probability of each class
	prob1 = 1
	probList1 = []
	for i in range(0, len(sample)):
		if sample[i] == 1:
			probList1.append(feat_probs1[i])
		else:
			probList1.append(1-feat_probs1[i])
	for i in range(0, len(probList1)):
		prob1 = prob1 * probList1[i]
	prob2 = 1
	prob_list2 = []
	for i in range(0, len(sample)):
		if sample[i] == 1:
			prob_list2.append(feat_probs2[i])
		else:
			prob_list2.append(1-feat_probs2[i])
	for i in range(0, len(prob_list2)):
		prob2 = prob2 * prob_list2[i]

	# Return class with higher probability
	if prob1 > prob2:
		return 1
	else:
		return 2

# Guess class using dependent Bayesian classification
def dpnt_guess_class_from_sample(tree1, tree2, sample):
	node_list1 = tree1.nodes(data = True)
	node_list2 = tree2.nodes(data = True)
	prob_list1 = []
	prob_list2 = []
	visited_nodes = []

	# Begin recursive tree operations
	recurs_dpnt_guess(tree1, get_root_node(node_list1), visited_nodes, sample, "root", prob_list1)
	total_prob1 = 1
	for prob in prob_list1:
		total_prob1 = total_prob1 * max(prob, 0.0000001)
	visited_nodes.clear()
	recurs_dpnt_guess(tree2, get_root_node(node_list2), visited_nodes, sample, "root", prob_list2)
	total_prob2 = 1
	for prob in prob_list2:
		total_prob2 = total_prob2 * max(prob, 0.0000001)

	if total_prob1 > total_prob2:
		return 1
	else:
		return 2

# Return the name of the root node in the dependence tree
def get_root_node(node_list):
	for n in node_list:
		if 'root' in n[1].keys():
			return n[0]
	print("Did not find root")
	return 1

# Calculate probability of class containing sample based on estimated dependence tree
def recurs_dpnt_guess(tree, n, visited_nodes, sample, parent_value, prob_list):
	# Only continue if node has not been checked
	if n not in visited_nodes:
		visited_nodes.append(n)
		# Handle root case
		if parent_value == "root":
			# Get the node's value from the sample
			node_val = sample[int(n)-1]
			# Assign correct probability
			if node_val == 1:
				prob_list.append(tree.node[n]['pr11'])
			else:
				prob_list.append(1 - tree.node[n]['pr10'])

			# Recurse through children of root node
			children = tree[n]
			for child in children:
				if isinstance(child, int):
					recurs_dpnt_guess(tree, child, visited_nodes, sample, n, prob_list)
		else:
			# Retrieve appropriate probabilities for all non-root nodes
			node_val = sample[int(n)-1]
			if node_val == 0 and sample[int(parent_value)-1] == 0:
				prob_list.append(1 - tree.node[n]['pr10'])
			elif node_val == 0 and sample[int(parent_value)-1] == 1:
				prob_list.append(1 - tree.node[n]['pr11'])
			elif node_val == 1 and sample[int(parent_value)-1] == 0:
				prob_list.append(tree.node[n]['pr10'])
			elif node_val == 1 and sample[int(parent_value)-1] == 1:
				prob_list.append(tree.node[n]['pr11'])

			# Recurse through children of this node
			children = tree[n]
			for child in children:
				if isinstance(child, int):
					recurs_dpnt_guess(tree, child, visited_nodes, sample, n, prob_list)


# Return a list of probabilities of value 1 occuring
# in each feature for a given dataset
def get_feature_probs(dataset):
	feature_occ = [0] * len(dataset[0])
	for state in dataset:
		for i in range(0, len(dataset[0])):
			if state[i] == 1:
				feature_occ[i] += 1
	for i in range(0, len(feature_occ)):
		feature_occ[i] = feature_occ[i] / len(dataset)
	return feature_occ

# Return list of averaged feature probabilities
def get_prob_avgs(feat_probs):
	avg_probs = [0] * len(feat_probs[0])
	for fold in feat_probs:
		for i in range(0, len(feat_probs[0])):
			avg_probs[i] += fold[i]
	for i in range(0, len(avg_probs)):
		avg_probs[i] = avg_probs[i] / len(feat_probs)
	return avg_probs


# Files to be read from
files = ['class1.csv', 'class2.csv', 'class3.csv', 'class4.csv']
full_set = create_dataset(files)
class_datasets = get_datasets(files)
print("\nIndependent classifier results: ")
k_fold_validation(full_set, class_datasets, 5, "independent", False)

print("\nDependent classifier results: ")
k_fold_validation(full_set, class_datasets, 5, "dependent", True)
