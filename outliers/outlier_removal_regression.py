
import random
import numpy
import matplotlib.pyplot as plt
import pickle

from outlier_cleaner import outlierCleaner


### load up some practice data with outliers in it
ages = pickle.load( open("practice_outliers_ages.pkl", "r") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "r") )


### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)



from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(ages_train, net_worths_train)

plt.plot(ages, reg.predict(ages), color='blue')

# plot all points in blue
plt.scatter(ages, net_worths, color='blue')

# plot the current training points in orange
plt.scatter(ages_train, net_worths_train, color='orange')

### identify and remove the most outlier-y points
predictions = reg.predict(ages_train)
cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )



### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    
    # the non-outlier ages, net worths, and errors
    ages_train2, net_worths_train2, errors_train2 = zip(*cleaned_data)
    ages_train2       = numpy.reshape( numpy.array(ages_train2), (len(ages_train2), 1))
    net_worths_train2 = numpy.reshape( numpy.array(net_worths_train2), (len(net_worths_train2), 1))

    # refit the cleaned data
    reg2 = linear_model.LinearRegression()
    reg2.fit(ages_train2, net_worths_train2)
    plt.plot(ages, reg2.predict(ages), color='red')
    plt.scatter(ages_train2, net_worths_train2, color='red')
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()

else:
    print "outlierCleaner() is returning an empty list, no refitting to be done"
    
    
print reg2.coef_[0][0]
print reg2.score(ages_test,net_worths_test)
