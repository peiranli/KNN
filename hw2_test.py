import numpy as np
import math
# Load data set and code labels as 0 ='NO', 1='DH', 2='SL'
labels = ['NO','DH','SL']
data = np.loadtxt('column_3C.dat',converters={6: lambda s: labels.index(s)})
# Separate features from labels
x = data[:, 0:6]
y = data[:, 6]
#Divide into training and test set
training_indices = range(0,80)+range(100,148)+range(160,280)
test_indices = range(80,100)+range(148,160)+range(280,310)
trainx = x[training_indices, :]
trainy = y[training_indices]
testx = x[test_indices, :]
testy = y[test_indices]

def L1_distance(x1,x2):
    distance = 0
    for index in range(0,6):
        distance_i = abs(x1[index]-x2[index])
        distance += distance_i
    return distance

def euclideanDist(x, xi):
    d = 0.0
    for i in range(len(x)-1):
        d += pow((float(x[i])-float(xi[i])),2)  #euclidean distance
    d = math.sqrt(d)
    return d

#KNN prediction and model training
def nn_predict(test_data, train_data):
    for i in test_data:
        dist = enclideanDist()
        for j in train_data:
            eu_dist = enclideanDist(i,j)
            
 
#Accuracy calculation function
def accuracy(test_data):
    correct = 0
    for i in test_data:
        if i[5] == i[6]:
            correct += 1
    accuracy = float(correct)/len(test_data) *100  #accuracy 
    return accuracy

nn_predict(testx, trainx)   
print test_dataset
print "Accuracy : ",accuracy(testx)

