#! python2
#!/usr/bin/python2

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
predictions = clf.predict(features_test)
t2 = time()
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, predictions)
print(accuracy)
print("Time to train data: %f" %round((t1-t0),3))
print("Time to predict new x: %f" %round((t2-t1),3))

#########################################################


