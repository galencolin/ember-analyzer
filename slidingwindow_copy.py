import jsonlines as json
import csv
import random
import ember
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import os
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import ember_init
import features

# Parse command-line arguments, like "file=xxx.arff"
args = {}
for arg in sys.argv:
	if ('=' in arg):
		splitted = arg.split('=', 1)
		if (len(splitted) == 2): # sanity check
			args[splitted[0]] = splitted[1]

np.set_printoptions(threshold=sys.maxsize)
size = 10000
RUN_ID = str(datetime.datetime.now()).split(".")[0].replace(":", "꞉") + ' (slidingwindow)' # can't have colons in filenames, but we can have this unicode thing
num_splits = 12
save = True
window_sizes = [2, 3, 5]
start_pos = 4
	
if ('save' in args):
	save = (args['save'].lower() != 'false')
	
if ('start_pos' in args):
	start_pos = int(args['start_pos'])

x_vals = []
y_vals = []

# put some zeros to the left of [val] (for the filenames and months)
def zero_pad(val, length = 2):
	st = str(val)
	while (len(st) < length):
		st = '0' + st
	return st

extractor = features.PEFeatureExtractor()

def read_file(filepath, shape):
	mem = np.memmap(filepath, dtype = np.float32, mode = "r", shape = shape)
	return mem
	
def read_files():
	for month in range(1, 13):
		filepath = 'months/x_vec_2018-' + zero_pad(month) + '.dat'
		vals = read_file(filepath, (size, extractor.dim))
		x_vals.append(vals)
				
		filepath = 'months/y_vec_2018-' + zero_pad(month) + '.dat'
		vals = read_file(filepath, size)
		y_vals.append(vals)

def get_split(data, start, end): # [start, end)
	return np.concatenate(data[start:end])
	
def eval_model(name, model, x, y):
	y_pred = model.predict(x)
	
	acc = sklearn.metrics.accuracy_score(y, y_pred)
	auc = sklearn.metrics.roc_auc_score(y, y_pred)
	rec = sklearn.metrics.recall_score(y, y_pred)
	pre = sklearn.metrics.precision_score(y, y_pred)
	
	print("Result for", name)
	print("Accuracy:", acc)
	print("AUC:", auc)
	print("Recall:", rec)
	print("Precision:", pre)
	print()
	
	return [acc, auc, rec, pre]
	
def build_classifiers(classifier_names, classifier_list, x, y):
	classifier_objects = []
	results = []
	for i in range(len(classifier_list)):
		print("Building classifier", classifier_names[i] + "...")
		object = classifier_list[i].fit(x, y)
		classifier_objects.append(object)
		
	return classifier_objects
	
def test_classifiers(classifier_names, classifier_list, x, y, label):
	results = []
	print("Testing results (label:", label + "):\n")
	
	for i in range(len(classifier_list)):
		print("Testing classifier", classifier_names[i] + "...")
		
		val = eval_model(classifier_names[i], classifier_list[i], x, y)
		
		results.append(val)
	return results

if __name__ == '__main__':
	read_files()
	
	print("Done reading.")
		
	names = [
	# "Nearest Neighbors", 
    # "Decision Tree", 
	"Random Forest", 
	# "AdaBoost",
	# "Neural Net", 
	# "QDA"
	]
	
	classifiers = [
    # KNeighborsClassifier(3),
    # DecisionTreeClassifier(),
    RandomForestClassifier(),
	# AdaBoostClassifier(),
    # MLPClassifier(alpha=1, max_iter=1000),
    # QuadraticDiscriminantAnalysis()
	]
	
	label_list = []
	result_list = []
	
	for i in range(start_pos, num_splits):
		print("Running testset", str(i + 1))
		cur_results = []
		for window_size in window_sizes:
			print("Running window", str(window_size))
			start_pos = max(0, i - window_size)
			x_train = get_split(x_vals, start_pos, i)
			y_train = get_split(y_vals, start_pos, i)
			classifier_list = build_classifiers(names, classifiers, x_train, y_train)
			result = test_classifiers(names, classifier_list, x_vals[i], y_vals[i], str(i + 1))[0]
			cur_results.append(result)
		label_list.append(str(i + 1))
		result_list.append(cur_results)
		
	model_accuracy = []
	model_auc = []
	model_recall = []
	model_precision = []
	for i in range(len(window_sizes)):
		results_accuracy = []
		results_auc = []
		results_recall = []
		results_precision = []
		for j in range(len(result_list)):
			results_accuracy.append(result_list[j][i][0])
			results_auc.append(result_list[j][i][1])
			results_recall.append(result_list[j][i][2])
			results_precision.append(result_list[j][i][3])
		model_accuracy.append(results_accuracy)
		model_auc.append(results_auc)
		model_recall.append(results_recall)
		model_precision.append(results_precision)
	
	print("Run ID:", RUN_ID)

	parent_dir = os.getcwd()
	run_dir = 'runs/' + RUN_ID
	if (save):
		os.mkdir(os.path.join(parent_dir, run_dir))
	
	# Plot the data
	plt.rcParams['xtick.labelsize'] = 20
	plt.rcParams['font.size'] = 25
	plt.rcParams['legend.fontsize'] = 18.75

	plt.figure(figsize = (13, 7))
	plt.title("Accuracy")
	plt.xlabel("Test group")
	for i in range(len(model_accuracy)):
		plt.plot(label_list, model_accuracy[i], label = window_sizes[i])
	plt.xticks(label_list, rotation=45, ha='right')
	plt.legend()
	# plt.tight_layout()
				
	if (save):
		plt.savefig(run_dir + "/" + "Accuracy (" + RUN_ID + ")" + ".png", bbox_inches='tight')
		
	plt.clf()

	plt.figure(figsize = (13, 7))
	plt.title("AUC")
	plt.xlabel("Test group")
	for i in range(len(model_auc)):
		plt.plot(label_list, model_auc[i], label = window_sizes[i])
	plt.xticks(label_list, rotation=45, ha='right')
	plt.legend()
	# plt.tight_layout()
				
	if (save):
		plt.savefig(run_dir + "/" + "AUC (" + RUN_ID + ")" + ".png", bbox_inches='tight')
		
	plt.clf()
	
	plt.figure(figsize = (13, 7))
	plt.title("Recall")
	plt.xlabel("Test group")
	for i in range(len(model_recall)):
		plt.plot(label_list, model_recall[i], label = window_sizes[i])
	plt.xticks(label_list, rotation=45, ha='right')
	plt.legend()
	# plt.tight_layout()
				
	if (save):
		plt.savefig(run_dir + "/" + "Recall (" + RUN_ID + ")" + ".png", bbox_inches='tight')
		
	plt.clf()
	
	plt.figure(figsize = (13, 7))
	plt.title("Precision")
	plt.xlabel("Test group")
	for i in range(len(model_precision)):
		plt.plot(label_list, model_precision[i], label = window_sizes[i])
	plt.xticks(label_list, rotation=45, ha='right')
	plt.legend()
	# plt.tight_layout()
				
	if (save):
		plt.savefig(run_dir + "/" + "Precision (" + RUN_ID + ")" + ".png", bbox_inches='tight')
		
	plt.clf()
		
	sys.exit(0)
	
	
	
	
	
	
	#######################################################################################################################################################
	#######################################################################################################################################################
	#######################################################################################################################################################
	
	
	
	
	
	
	
	data_dir = 'D:\ember_dataset_2018_2.tar\ember2018'
	np.random.seed(2424)
	
	emberdf = ember.read_metadata(data_dir)
	X_train, y_train, X_test, y_test = ember.read_vectorized_features(data_dir, feature_version=1)
	
	size = 100000
	X_sub = X_train[:size]
	y_sub = y_train[:size]
	X_test_sub = X_test[:size]
	y_test_sub = y_test[:size]
		
	print(size)
		
	print("done reading")
	
	names = [
	"Nearest Neighbors", 
	# "Linear SVM", 
	# "RBF SVM", 
	# "Gaussian Process",
    "Decision Tree", 
	"Random Forest", 
	# "Neural Net", 
	"AdaBoost",
    # "Naive Bayes", 
	# "QDA"
	]
	
	classifiers = [
    KNeighborsClassifier(3),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    # MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    # GaussianNB(),
    # QuadraticDiscriminantAnalysis()
	]
		
	for name, classifier in zip(names, classifiers):
		model = classifier
		model = model.fit(X_sub, y_sub)
		print(name, cross_val_score(model, X_test_sub, y_test_sub, cv = 10, scoring = 'accuracy'))
	
	sys.exit(0)
	
	dtree = tree.DecisionTreeClassifier()
	
	print("done creating model")
	
	dtree = dtree.fit(X_sub, y_sub)
	
	print("done fitting model")
	
	print(cross_val_score(dtree, X_test_sub, y_test_sub, cv = 10, scoring = 'accuracy'))
	
	print ("done evaluating model")
	
	sys.exit(0)

	lines = []
	with json.open('train_features_2.jsonl') as reader:
		for line in reader:
			if (line['label'] != -1):
				lines.append(line)

	# print(lines[0])
	# print(lines[0].keys())
	print(len(lines))

	random.seed(77)
	random.shuffle(lines)

	header = []
	for i in range(256):
		header.append("histogram" + str(i))
	for el in lines[0]['general'].keys():
		if (el != 'characteristics'):
			header.append(el)
	header.append('label')

	with open('histogram.csv', 'w', newline = '') as f:
		writer = csv.writer(f, delimiter = ',')
		writer.writerow(header)
		for line in lines[:10000]:
			dat = line['histogram']
			for el in line['general'].keys():
				if (el != 'characteristics'):
					dat.append(line['general'][el])
			dat.append(line['label'])
			writer.writerow(dat)