#!/usr/bin/python

import matplotlib.pyplot as plt
from operator import itemgetter
import sys
import pickle
import numpy as np
import random
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

def apply_SVC(X_train, label_train, X_test, label_test):
	print "Training data using SVC ..."
	from sklearn.svm import SVC
	from sklearn import grid_search
	
	X_train_norm = normalize(X_train)
	X_test_norm = normalize(X_test)
	parameters = {'C': [10000, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf'], 'random_state':[42]}

	svr = SVC()
	clf = grid_search.GridSearchCV(svr, parameters,scoring='recall')
	
	
	
	#clf = SVC(C= 100, kernel= 'sigmoid', random_state=42)
	
	clf.fit(X_train_norm, label_train)
	best_clf = clf.best_estimator_
	#print "Best Parameter for SVC using GridSearchCV", clf.grid_scores_ 
	pred_SVC = best_clf.predict(X_test_norm)
	print "accuracy of SVC", clf.score(X_test_norm, label_test)
	from sklearn.metrics import precision_score, recall_score, f1_score
	print "f1 score is", f1_score(label_test, pred_SVC)
	print "precision score is", precision_score(label_test, pred_SVC)
	print "recall score is", recall_score(label_test, pred_SVC)

	print pred_SVC
	return best_clf


def normalize(data):
	print type(data)
	print "Normalizing features for an array of shape: ", data.shape
	norm_data = np.zeros(shape=(data.shape[0], data.shape[1]))
	for j in range(data.shape[1]):
		maxi = data[:,j].max()
		mini = data[:,j].min()

		norm_data[:,j] = (data[:,j] - mini)/(float(maxi)-mini)

	return norm_data

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def apply_GridSearchCV(X_train, label_train, X_test, label_test):
	"Using GridSearchCV to do exhaustive search over specified parameter values for RandomForestClassifier."
	from sklearn import grid_search
	from sklearn.ensemble import RandomForestClassifier
	random.seed(42)
	parameters = {'n_estimators':[10, 15, 20, 25], 'min_samples_split':[2, 3, 4, 5], 'random_state':[42]}
	clf = grid_search.GridSearchCV(RandomForestClassifier(random_state=42), parameters, verbose=3, scoring = 'recall')
	clf.fit(X_train, label_train) 

	best_clf = clf.best_estimator_
	print "best edtimator", best_clf

	print "feature importances of RandomForestClassifier: ", best_clf.feature_importances_

	pred_RF = best_clf.predict(X_test)
	#print "List of the feature importances", clf.feature_importances_ 
	
	print "Accuracy of Random Forest", best_clf.score(X_test, label_test)
	from sklearn.metrics import precision_score, recall_score, f1_score
	print "F1 score is", f1_score(label_test, pred_RF)
	print "Precision score is", precision_score(label_test, pred_RF)
	print "Recall score is", recall_score(label_test, pred_RF)
	return best_clf

def apply_RF(X_train, label_train, X_test, label_test):
	# taking train and test data, do a random forest classification
	print "Training data using RandomForestClassifier..."
	from sklearn.ensemble import RandomForestClassifier
	
	clf= RandomForestClassifier(n_estimators=10, min_samples_split=2)
	
	clf.fit(X_train, label_train) 
	
	pred_RF = clf.predict(X_test)
	#print "List of the feature importances", clf.feature_importances_ 
	
	print "Accuracy of Random Forest", clf.score(X_test, label_test)
	from sklearn.metrics import precision_score, recall_score, f1_score
	print "F1 score is", f1_score(label_test, pred_RF)
	print "Precision score is", precision_score(label_test, pred_RF)
	print "Recall score is", recall_score(label_test, pred_RF)
	
	return clf

def produce_spliting_array(total_size, train_prop): #for sampling
	random.seed(3)
	
	split_array = []
	for i in range(total_size):
		if random.random() < train_prop:
			split_array.append(1)
		else:
			split_array.append(0)

	return np.array(split_array)

def transform_using_PCA(features):
	print "Using PCA to reduce dimentionalities ..."
	#using PCA, reduce dimentionalities while keeping .99 variations in features.
	from sklearn.decomposition import PCA
	n = 3
	sum_explained = 0
	while sum_explained < .99:
		n += 1 
		pca = PCA(n_components=n) #number of latent variables
		pca.fit(features)
		sum_explained = sum(pca.explained_variance_ratio_)


	
	return pca.fit_transform(features)

def plot_features(X, y):
	from matplotlib import pyplot
	pyplot.scatter(X, y)
	pyplot.show()

def anomaly_detection(features, labels):
	# In this function, I try to use anomaly detection method (using mutivariate gaussian distribution) to identify poi-s
	non_pois = features[labels==0]
	pois = features[labels==1]
	print "non poi size", non_pois.shape, pois.shape, features.shape

	## Spliting data to train, test and cross validation set for anomaly detection

	split1 = produce_spliting_array(non_pois.shape[0], .75 )
	X_train = non_pois[split1==1]

	X_intermediate = non_pois[split1==0]

	print "size intermediate", X_intermediate.shape

	split2 = produce_spliting_array(X_intermediate.shape[0], .5 )

	X_test = X_intermediate[split2==1]
	label_test = np.zeros((X_test.shape[0],), dtype=np.int) - 1

	X_cv = X_intermediate[split2==0]
	label_cv = np.zeros((X_cv.shape[0],), dtype=np.int) - 1

	split3 = produce_spliting_array(pois.shape[0], .5 )
	X_test = np.vstack((X_test, pois[split3==1]))
	label_test = np.hstack((label_test, np.ones(sum(split3), dtype=np.int)))

	X_cv = np.vstack((X_cv, pois[split3==0]))
	label_cv = np.hstack((label_cv, np.ones(sum(split3==0), dtype=np.int)))



	print "size X_train", X_train.shape
	print "size test data", X_test.shape, label_test.shape
	print "size cv data", X_cv.shape, label_cv.shape
	print "size splits", len(split1), len(split2), len(split3)

	from sklearn.covariance import EllipticEnvelope
	detector = EllipticEnvelope(contamination=.85)
	detector.fit(X_train)
	pred_cv = detector.predict(X_cv)
	print pred_cv
	print label_cv
	print detector.score(X_cv, label_cv)


### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
selected_features_list = ["poi", 'salary', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'expenses', 'deferred_income', 'long_term_incentive', 'shared_receipt_with_poi']
#all_features = ["poi", 
#'salary',  'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value', 
# 'expenses', 'loan_advances', 'director_fees', 'deferred_income', 'long_term_incentive',
#'shared_receipt_with_poi',  'to_messages','from_messages', 'other', 'from_this_person_to_poi',  'from_poi_to_this_person']
all_features = ["poi", 
				'salary', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'expenses', 'deferred_income', 'long_term_incentive',
				'deferral_payments', 'restricted_stock_deferred', 'total_stock_value',  'loan_advances', 'director_fees',
				'to_messages',  'from_messages', 'from_this_person_to_poi',  'from_poi_to_this_person', 'shared_receipt_with_poi']


### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

## print all features in the dataset



### we suggest removing any outliers before proceeding further

data_dict.pop("TOTAL") # this is outlier
data_dict.pop("LOCKHART EUGENE E") #this is outlier because all the columns are zero
### if you are creating any new features, you might want to do that here
### store to my_dataset for easy export below
my_dataset = data_dict

'''
counter = 0
for item in my_dataset:
	counter += 1
	print item, my_dataset[item]
	if counter >10:
		break
'''
### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, all_features, remove_NaN=True, remove_all_zeroes=True )
print "Total number of data points are: ", data.shape[0]
print "Allocation across classes (POI/non-POI): (%d, %d)" %(sum(data[:,[0]] == 1), sum(data[:,[0]] == 0))
print "Total number of features in the raw dataset are:", len(my_dataset[my_dataset.keys()[0]])


#email = featureFormat(my_dataset, ['email_address'], remove_NaN=False)

'''
# email email_address of pois
for key in my_dataset:
	if my_dataset[key]['poi'] == 1:
		print my_dataset[key]['email_address']
'''
### if you are creating new features, could also do that here
# create two new variables : to_poi/to_messages and from_poi/from_messages
#print "multiply receive and send", sum(data[:,11] * data[:,12] == 0)

np.seterr(divide='ignore', invalid='ignore')
to_poi_prop = np.array(data[:,16]/data[:,14])
to_poi_prop[data[:,14]==0] = 0
from_poi_prop = data[:,17]/data[:,15]
from_poi_prop[data[:,15]==0] = 0

#data = data[:,[0,1,2,3,4,5,6,7,8,13]]
#financial_data = data[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]]
financial_data = data[:,[1,4,7,10,11]]
#PCA
financial_data_transformed = transform_using_PCA(financial_data)
print "Reduce dimentionalities of financial data from %d --> %d" %(13, financial_data_transformed.shape[1])

data = np.hstack((data[:,[0]], financial_data, data[:,[-1]], to_poi_prop[np.newaxis,:].T, from_poi_prop[np.newaxis,:].T))

#updating my_dataset and selected featurelist

selected_features_list = ['poi', 'from_poi_prop', 'to_poi_prop', 'shared_receipt_with_poi',
							'salary', 'bonus', 'total_stock_value', 'restricted_stock_deferred', 'total_payments']

row_number = 0
for name in data_dict:
	my_dataset[name]['from_poi_prop'] = from_poi_prop[row_number]
	my_dataset[name]['to_poi_prop'] = to_poi_prop[row_number]
	
	row_number +=1



print "Total number of features used in this project are: ", len(selected_features_list)
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)
features = np.array(features)
labels = np.array(labels)


### machine learning goes here!
### please name your classifier clf for easy export below

# here I used different features to plot data and get sense of data
#plot_features(data[:,1], financial_data_transformed[:,0]) #plot salary vs first PC of financial features


## Spliting data to train, test 
from sklearn.cross_validation import train_test_split
X_train, X_test, label_train, label_test = train_test_split(features, labels, test_size=0.3, random_state=2015)


### get rid of this line!  just here to keep code from crashing out-of-box
print "Number of features to use in Machine Learning", data.shape[1]
# Random Forest
'''
# Using k-fold to divide data to training and testing datasets
from sklearn import cross_validation
kf = cross_validation.KFold(features.shape[0], n_folds=2, random_state=42)


for train_index, test_index in kf:
	X_train, X_test = features[train_index], features[test_index]
	label_train, label_test = labels[train_index], labels[test_index]
'''

#clf = apply_RF(X_train, label_train, X_test, label_test)
clf = apply_GridSearchCV(X_train, label_train, X_test, label_test)

#clf = apply_SVC(X_train, label_train, X_test, label_test)
















### dump your classifier, dataset and features_list so 
### anyone can run/check your results
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(selected_features_list, open("my_feature_list.pkl", "w") )



