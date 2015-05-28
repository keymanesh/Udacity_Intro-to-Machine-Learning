# Udacity_ Intro to Machine Learning

## Overview of data

Enron had so many employees and only some of them were involved in the scandal. We already know fraction of those who were involved but we don’t know all of them so here in this project we are trying to employ machine learning techniques to find out who else were possibly involved in this scandal. 

* Total number of data points are:  144

* Allocation across classes (POI/non-POI): (18, 126)

* Total number of features in the raw dataset are: 21

* Total number of features used in this project are:  9

There were couple of outliers in the data. An obvious one which is named ‘TOTAL’, and I removed it. Also there was another item whose features were all equal to zero ("LOCKHART EUGENE E"), 
I removed that as well. Based on my outlier detection method I could find some other candidates for outlier but I decided to keep them as they were known Person-Of-Interests(POIs) so keeping them may help the classification. To find outliers I simply used scatter plot on different axis of data for example, ‘salary’ vs ‘bonus’; I also used dimensionality reduction method (PCA) to be able to plot data using more information. More specifically I used ‘salary’ vs first transformed factor of other financial data: some of the data point that were candidate for outliers:

* Lay, Kenneth

* Skilling, Jeffrey

### Features in the dataset

Features in this dataset fall into three major types, namely **financial features**, **email features** and **POI labels**.

financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']  (all units are in US dollars)

email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)

POI label: [‘poi’] (boolean, represented as integer)

The last one (POI label) is an output variable. Since the scale for the first two variables are very different, feature scaling seems to be important. 
However in this project I used RandomForest algorithm in which feature scaling doesn’t impact much. 

Also, using four features in the dataset, I defined two new features for my classification and I used them instead of four contributing factors:

* From_poi_to_person / from_messages

* From_person_to_poi / to_messages

The reason I used these two new features were:

* Similar to feature scaling, it makes different features to be in similar scales for my classifications 

* They are better indication of to what extend a specific person were involved with POIs.

Here is the list of all features that I used:

selected_features_list = ['poi', 'from_poi_prop', 'to_poi_prop', 'shared_receipt_with_poi',
'salary', 'bonus', 'total_stock_value', 'restricted_stock_deferred', 'total_payments']

The first one is the label, so I used 8 different features to train RF algorithm.

feature importances of RandomForestClassifier:  [ 0.18132923  0.17936594  0.09689551  0.00039633  0.17599177  0.17719354  0.07310256  0.11572512]

I used these features mostly by hand, I used different combinations to test out which features work out best and I ended up using the 8 above features.

I did following things to come up with best features:

* I studied sample of data to understand all the features

* I tried different combination of features on train and cv dataset to see what combination gives a better score.

* I divided features into three main categories of payment, stock and email info

* From each category I chose the ones that worked the best with the score of the CV dataset for my algorithm and also capture the most information. For example, by looking at data one can obviously see that bonus is a good candidate while loan advances is not as the former has very wide range of numbers and the latter is NaN for more than 90% of cases!

### Algorithm

After examining different algorithms, I ended up using Random Forest which gave me the best results. Some techniques that I used:

* Anomaly detection using multivariate Gaussian distribution

* SVC

* RandomForest

### Tuning Algorithm

To tune an algorithm is to set the parameters of an algorithm in a way that you can get the best result out of the algorithm. Even the best algorithm if has not been tuned properly may work very
 poorly so it is important to know different parameters of machine learning algorithms and tune them properly so in our application it works efficiently.

We can divide data to 3 folds of train, cross validation and test data. We can train our classifier using train data. Tune the parameters using cross validation data set and then finally report the performance using test data.

For my Random Forest Classifier I used grid_searchCV in which I set the parameters and train the data using different set of parameters:

parameters = {'n_estimators':[5, 10, 15, 20, 25], 'min_samples_split':[2, 3, 4, 5]}

best edtimator RandomForestClassifier(bootstrap=True, compute_importances=None,
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
            min_samples_split=2, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=42, verbose=0)


### Validation

Typical scenario for many machine learning techniques is to divide data to train, cross validation and test dataset. Using train dataset you can train your algorithm, using cross validation dataset you can tune your algorithm and test set is only to report performance.


### Evaluation

Accuracy: it simply says to what extend the predictor could predict the labels correctly. However this metric is not very effective specially for case of skewed data.

Precision: it says what percentage of the data points that has been classified as poi, are really poi.

Recall: it says what percentage of POIs was predicted (classified) correctly by our algorithm.
 
Following is the performance metrics for my Random Forest Classifier:

*accuracy*: 0.909

*f1 score*: 0.5

*precision score*: 0.5

*recall score*: 0.5
