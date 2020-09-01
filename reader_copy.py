import json as real_json
import jsonlines as json
import csv
import random
import ember
import sys
import ember_init
import numpy as np
import features

np.set_printoptions(threshold=sys.maxsize)
size = 50000
data_dir = 'months_' + str(size)

# run this for all train_features_[1-5] and test_features, or just be smart and do it all at once
def create_subsets(filename):
	months = set()
	lines = []
	with json.open(filename) as reader:
		for line in reader:
			if (line['label'] != -1):
				lines.append(line)
				months.add(line['appeared'])
	
	bymonth = {}
	for x in months:
		bymonth[x] = []
	
	for line in lines:
		bymonth[line['appeared']].append(line)

	print(lines[0])
	# print(lines[0].keys())
	print(len(lines))

	random.seed(77)
	# random.shuffle(lines)
	
	print(bymonth.keys())
		
	for month in bymonth.keys():
		arr = bymonth[month]
		random.shuffle(arr)
		sliced = arr[:size]
		with json.open(data_dir + '/' + 'data_' + month + '.jsonl', mode='w') as writer:
			for i in range(size):
				writer.write(sliced[i])
				if (i % 100 == 0):
					print("Month:", month + ",", "line:", i + 1)

def get_vec(header):
	return ember_init.vectorize_without_forced_file_reads(header)

# put some zeros to the left of [val] (for the filenames and months)
def zero_pad(val, length = 2):
	st = str(val)
	while (len(st) < length):
		st = '0' + st
	return st
	
def create_train():
	extractor = features.PEFeatureExtractor()
	
	for month in range(1, 13):
		inpath = data_dir + '/data_2018-' + zero_pad(month) + '.jsonl'
		outpath = data_dir + '/x_vec_2018-' + zero_pad(month) + '.dat'
		
		lines = []
		with open(inpath, 'r') as f:
			for line in f.readlines():
				lines.append(line[:-1].strip())
		
		for i in range(len(lines)):
			if (i % 100 == 0):
				print("Month:", zero_pad(month) + ",", "line:", i + 1)
			lines[i] = get_vec(lines[i])
			
		nrows = len(lines)
			
		mem = np.memmap(outpath, dtype = np.float32, mode = "w+", shape = (nrows, extractor.dim))
		
		for i in range(len(lines)):
			mem[i] = lines[i]
			
def create_test():
	extractor = features.PEFeatureExtractor()
	
	for month in range(1, 13):
		inpath = data_dir + '/data_2018-' + zero_pad(month) + '.jsonl'
		outpath = data_dir + '/y_vec_2018-' + zero_pad(month) + '.dat'
		
		lines = []
		with open(inpath, 'r') as f:
			for line in f.readlines():
				lines.append(line[:-1].strip())
		
		for i in range(len(lines)):
			if (i % 100 == 0):
				print("Month:", zero_pad(month) + ",", "line:", i + 1)
			lines[i] = real_json.loads(lines[i])['label']
			
		nrows = len(lines)
			
		mem = np.memmap(outpath, dtype = np.float32, mode = "w+", shape = nrows)
		
		for i in range(len(lines)):
			mem[i] = lines[i]

if __name__ == '__main__':
	print("Creating subsets")
	filenames = [
	'train_features_1.jsonl', 
	'train_features_2.jsonl', 
	'train_features_3.jsonl', 
	'train_features_4.jsonl', 
	'train_features_5.jsonl', 
	'test_features.jsonl'
	]
	for filename in filenames:
		print("Creating subsets for file", filename)
		create_subsets(filename)
	print("Creating training sets")
	create_train()
	print("Creating testing sets")
	create_test()
	
	exit(0)

	header = []
	for i in range(256):
		header.append("histogram" + str(i))
	for el in lines[0]['general'].keys():
		if (el != 'characteristics'):
			header.append(el)
	header.append('label')

	with open('test_histogram.csv', 'w', newline = '') as f:
		writer = csv.writer(f, delimiter = ',')
		writer.writerow(header)
		for line in lines[:10000]:
			dat = line['histogram']
			for el in line['general'].keys():
				if (el != 'characteristics'):
					dat.append(line['general'][el])
			dat.append(line['label'])
			writer.writerow(dat)