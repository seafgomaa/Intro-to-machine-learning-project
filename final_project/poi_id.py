#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
'''ÇÎÊíÇÑ ãÌãæÚå İíÊÔÑÒ ÍÓÈ ÇåãíÊåÇ İí ÎÏãÉ ÇáÔÛá ÍÓÈ æÌåÉ äÙÑß'''
features_list = ['poi','salary','to_messages','total_payments','shared_receipt_with_poi','total_stock_value','from_messages','from_this_person_to_poi','from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


''' ØÈÇÚÉ ãÍÊæíÇÊ ÇáÏÇÊÇ ÏíßÊ ÈØÑíŞÊíä '''
import json
#print json.dumps(data_dict, indent = 4)


##for person, features in data_dict.iteritems():
##         print features




### Task 2: Remove outliers

'ãä ÏÑÓ ÇáÇæÊáÇíÑ íÊã Úãá ãÚÇÏáÉ ÑíÌÑÔä Èíä ÇáÓÇáÑí æÇáÈæäÕ æÇÈÚÏ Şíã Úä ÎØ ÇáÑíÌÑÔä Êã ÇÚÊÈÇÑåÇ ÇæÊáÇíÑ'
data_dict.pop('LAY KENNETH L')
data_dict.pop('SKILLING JEFFREY K')
data_dict.pop('TOTAL')

'''ÑÓã äŞØ ßá İíÊÔÑ ãÚ ÇáÇÎÑ Úáí ÇÍÏÇËíÇÊ ÇÍÇÏíÉ ÇáÈÚÏ'''
##data = featureFormat(data_dict, features_list)
##def visualization(data, a, b, a_name, b_name):
##
##    f1 = []
##    f2 = []
##    y = []
##    for point in data:
##        f1.append(point[a])
##        f2.append(point[b])
##        c = 'red' if point[0]==True else 'blue'
##        y.append(c)
##    plt.scatter(f1, f2, c=y)
##    
##    
##    plt.xlabel(a_name)
##    plt.ylabel(b_name)
##    plt.show()
##    
##
##
##def visualize():
##    
##    for i in range(2,len(features_list)):
##        visualization(data, 1, i, 'salary', features_list[i])
##       
##   
##visualize()

'  ÇáÍÕæá Úáí ÇáÇæÊáÇíÑ ÈØÑíŞÊíä ÇáãíÏíÇä æÇáÈÑíÓäÊíá ãä ãÕİæİå Şíã'
##def mad_based_outlier(points, thresh=20):
##    median = np.median(points, axis=0)
##    
##    diff = (points - median)**2
##   
##    diff = np.sqrt(diff)
##    
##    med_abs_deviation = np.median(diff)
##    
##    modified_z_score = 0.6745 * diff / med_abs_deviation
##    
##    return modified_z_score > thresh
##
##def percentile_based_outlier(data, threshold=50):
##    diff = (100 - threshold) / 2.0
##    minval, maxval = np.percentile(data, [diff, 100 - diff])
##    return (data < minval) | (data > maxval)
##
##for i in range(len(features_list)):
##    f1=[]
##    for point in data:
##        f1.append(point[i])
##    j=i-1    
##    x=[]
##    t=[]
##    x=(mad_based_outlier(f1))
##    z=zip(f1,x)
##    for i in range(len(f1)):
##        if z[i][1]==True:
##            t.append(z[i][0])
##    print 'out of ',features_list[j],':' ,t




### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


for i in data_dict:
     data_dict[i]['from_poi_percent']= float(data_dict[i]['from_poi_to_this_person']) / float(data_dict[i]['to_messages'])
     data_dict[i]['to_poi_percent']= float(data_dict[i]['from_this_person_to_poi']) / float(data_dict[i]['from_messages'])

my_dataset = data_dict
#print json.dumps(my_dataset, indent = 4)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(min_weight_fraction_leaf=.2)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print 'Decision Tree accuracy ', accuracy_score(pred, labels_test)

##print precision_score(pred, labels_test)
##
##print recall_score(pred, labels_test)





### Task 5: Tune your classifier to achieve better than .3 precision and recall 

### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
print 'Decision Tree accuracy at .3 test ', accuracy_score(pred, labels_test)

'ãÚÑİÉ ÇáİíÊÔÑ ÇáÇåã æÇáÇßËÑ ÊÃËíÑÇ Úáí ÌæÏÉ ÇÏÇÁ ÇáßáÇÓíİíÑ'
x= clf.feature_importances_
y=[]
t=[]
for i in range(len(x)):
    if x[i]> 0 :
        y.append(i)
        t.append(x[i])
z=zip(features_list,t)
z.sort(reverse=True,key=lambda t: t[1])
print z
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.





dump_classifier_and_data(clf, my_dataset, features_list)
