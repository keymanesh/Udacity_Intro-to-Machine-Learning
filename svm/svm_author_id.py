#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
print("original size of features are ", len(features_train))
#features_train = features_train[:round(len(features_train)/100)]
#labels_train = labels_train[:round(len(labels_train)/100)] 
from sklearn.svm import SVC
print(1)
clf = SVC(kernel="rbf", C=10000)
print(2)
clf.fit(features_train, labels_train)
print(3)
pred = clf.predict(features_test)
print(4)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
counter = 0
for pre in pred:
	if pre==1:
		counter += 1

print("prediction for chris ", counter)

print("accuracy is ", accuracy)

#########################################################


