import csv

def process_dataset(file_name):
	dat = None
	with open(file_name, 'r') as f:
		# Convert every field in CSV file to integers
		dat = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]
		# Iterate over every row
		for i in range(1, len(dat[0])):
			convert_col_binary(dat, i)
	with open('bin_wine.csv', 'w', newline='') as f:
		output = csv.writer(f)
		output.writerows(dat)
		
def convert_col_binary(dataset, column_index):
	total_val = 0
	for row in dataset:
		total_val += row[column_index]
	avg = total_val / len(dataset)
	for row in dataset:
		if row[column_index] > avg:
			row[column_index] = 1
		else:
			row[column_index] = 0

def seperate_classes(file_name, num_classes):
	classes = [[] for x in range(num_classes)]
	with open(file_name, 'r') as f:
		dat = [list(map(float,rec)) for rec in csv.reader(f, delimiter=',')]
		for row in dat:
			conv_row_to_int(row)
			classes[int(row[0])-1].append(row[1:])
	for i in range(0, len(classes)):
		with open('class' + str(i+1) + '.csv', 'w', newline='') as f:
			output = csv.writer(f)
			output.writerows(classes[i])

def conv_row_to_int(row):
	for i in range(0, len(row)):
		row[i] = int(row[i])

process_dataset('wine.csv')
seperate_classes('bin_wine.csv', 3)