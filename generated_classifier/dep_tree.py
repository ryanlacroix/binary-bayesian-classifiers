# Build a dependence tree to generate data from
# Generated data will be used to develop classifiers

import random

# Recursively generate dependent values into a master sample
def recurs_get(node, master_sample):
	roll = random.random()
	# If this is the root node
	if node.parent == "root":
		# Set the independent feature based on roll
		if roll > node.pr10:
			master_sample[node.featIndex - 1] = 1
		else:
			master_sample[node.featIndex - 1] = 0
	else:
		# This is not the root node
		if master_sample[node.parent.featIndex - 1] == 0:
			# Parent feature is 0
			if roll > node.pr10:
				master_sample[node.featIndex - 1] = 1
			else:
				master_sample[node.featIndex - 1] = 0
		else:
			# Parent feature is 1
			if roll > node.pr11:
				master_sample[node.featIndex - 1] = 1
			else:
				master_sample[node.featIndex - 1] = 0

	# Recurse through children nodes
	for child in node.childList:
		recurs_get(child, master_sample)

class DependenceTree:
	def __init__(self):
		self.root = predefined_dep_tree()

	def get_sample(self):
		sample = [None] * 10
		recurs_get(self.root, sample)
		return sample

# Node in the dependence tree
class Node:
	def __init__(self, parent, featIndex):
		# Generate random values to assign to 
		# Probability of 1 given parent is 0
		self.pr10 = random.random()
		# Probability of 1 given parent is 1
		self.pr11 = random.random()
		self.parent = parent
		self.featIndex = featIndex
		self.childList = []
		# Add self to parent's children
		if (parent != "root"):
			parent.add_child(self)
		
	def add_child(self, node):
		self.childList.append(node)

# Return root node of dependence tree with filled in probabilities
def predefined_dep_tree():
	# Root is node5
	node5 = Node("root", 5)

	node3 = Node(node5, 3)
	node8 = Node(node3, 8)
	node9 = Node(node8, 9)
	node10 = Node(node9, 10)

	node7 = Node(node5, 7)
	node2 = Node(node7, 2)
	node1 = Node(node2, 1)

	node4 = Node(node1, 4)
	node6 = Node(node1, 6)

	return node5