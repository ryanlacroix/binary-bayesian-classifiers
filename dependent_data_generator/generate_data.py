# Generate some data based on dependence trees created for each class

from dep_tree import *
import csv
import sys

def generate_datasets(num_classes, size):
	# Iterate through classes
	for i in range(0, num_classes):
		# Create the dependence tree for this class
		class_tree = DependenceTree()
		# Open a new CSV file to store samples for this class
		filename = 'class' + str(i + 1) + '.csv'
		with open(filename, 'w', newline='') as f:
			output = csv.writer(f)
			# Create each sample
			for j in range(0,size):
				sample = class_tree.get_sample()
				output.writerow(sample)

# Generate data for given number of classes.
# Each set is based on the same dependence tree populated
# with different probabilities.
generate_datasets(4, 2000)