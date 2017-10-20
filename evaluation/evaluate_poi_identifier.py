#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import numpy as np
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



from sklearn import cross_validation
train_features,test_features,train_labels,test_labels =cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)

### it's all yours from here forward!  
from sklearn import tree
from sklearn.metrics import accuracy_score
clf= tree.DecisionTreeClassifier()
clf= clf.fit(train_features,train_labels)
pred=clf.predict(test_features)


print np.count_nonzero(pred)
print len(test_features)
k=[0]*29
print accuracy_score(k,test_labels)

print pred,test_labels

from sklearn.metrics import confusion_matrix, precision_score, recall_score

print precision_score(pred,test_labels)

print recall_score(pred,test_labels)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

m=confusion_matrix(true_labels,predictions)
print m[1][1]
print m[1][0]
print m[0][1]
print m[0][0]



print precision_score(predictions,true_labels)

print recall_score(predictions,true_labels)























































































































































































































































