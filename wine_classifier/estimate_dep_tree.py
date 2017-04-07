import math
import csv
import networkx as nx
import matplotlib.pyplot as plt
from max_span_tree import *

# Make an estimated dependence tree based on a dataset
def estimate_dep_tree(dataset, num_features, show_tree = False):
	# List of all possible states of two given nodes
	states = []
	for i in range(0, 2):
		for j in range(0, 2):
			states.append((i, j))

	# List of every pair of features (edges)
	feature_combos = []
	for i in range(0,num_features):
		for j in range(0, num_features):
			if (i, j) and (j, i) not in feature_combos and i != j:
				feature_combos.append((i, j))
	
	# Generate weights for each edge
	graph = []
	for edge in feature_combos:
		total_weight = 0
		for state in states:
			# get probability of this state occurring from the dataset
			pr_vi_vj = get_state_probability(dataset, edge, state)
			pr_vi = get_ind_probability(dataset, edge[0], state[0])
			pr_vj = get_ind_probability(dataset, edge[1], state[1])
			
			# Protect against log errors
			if pr_vi_vj <= 0:
				pr_vi_vj = 0.000001
			if pr_vi <= 0:
				pr_vi = 0.000001
			if pr_vj <= 0:
				pr_vj = 0.000001

			mutual_information = pr_vi_vj*math.log(pr_vi_vj/(pr_vi*pr_vj),2)
			total_weight += mutual_information
		graph.append((edge, total_weight))

	# By this point we have a fully connected graph with weights
	dep_tree = MaxSpanTree(graph)
	if show_tree == True:
		nx.draw_circular(dep_tree.tree, with_labels=True)
		plt.show()
	return dep_tree.tree

# Populate the tree with state probabilities
# Takes in a networkx Graph
def build_tree_probs(tree, dataset):
	node_list = tree.nodes()
	visited_nodes = []
	recurs_node_probs(tree, node_list[0], dataset, visited_nodes, "root")

# Estimate probabilities in dependence tree node based on data set
def recurs_node_probs(tree, n, data_subset, visited_nodes, parent_value):
	# Only continue if node has not already been updated
	if n not in visited_nodes:
		visited_nodes.append(n)
		if parent_value != "root":
			# Set values based on parent value
			tree.node[n]['pr10'] = get_state_probability(data_subset, (int(n)-1,int(parent_value)-1), (1,0))
			tree.node[n]['pr11'] = get_state_probability(data_subset, (int(n)-1,int(parent_value)-1), (1,1))
		else:
			tree.node[n]['pr10'] = get_ind_probability(data_subset, int(n)-1, 1)
			tree.node[n]['pr11'] = get_ind_probability(data_subset, int(n)-1, 1)
		
		children = tree[n]
		for child in children:
			if isinstance(child, int):
				recurs_node_probs(tree, child, data_subset, visited_nodes, n)
			
		children = tree[n]
		for child in children:
			if isinstance(child, int):
				recurs_node_probs(tree, child, data_subset, visited_nodes, n)

# Splits a dataset into two:
# new_subset[0] where feature is always 0
# new_subset[1] where feature is always 1
def split_dataset(feature, dataset):
	new_subset = [[] for x in range(2)]
	for row in dataset:
		if row[feature] == 1:
			new_subset[1].append(row)
		else:
			new_subset[0].append(row)
	return new_subset

# Return probability of individual state occurring on an edge
def get_state_probability(dataset, edge, state):
	# Keep track of total occurrences
	total_occ = 0
	# Rows to consider
	feat_1 = edge[0]
	feat_2 = edge[1]

	for row in dataset:
		if (row[feat_1], row[feat_2]) == state:
			total_occ += 1

	probability = total_occ / len(dataset)
	return probability

# Return probability of individual state occurring on a feature
# Note: state is a single value in this case
def get_ind_probability(dataset, feature, state, testing = False):
	total_occ = 0
	for row in dataset:
		if row[feature] == state:
			total_occ += 1

	probability = total_occ / max(len(dataset), 1)
	return probability