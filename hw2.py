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

def L2_distance(x1,x2):
    distance = 0
    for index in range(0,6):
        distance_i = math.pow((x1[index]-x2[index]),2)
        distance += distance_i
    return math.sqrt(distance)

def create_result_vector(length):
    result = []
    for i in range(length):
        result.append(-1)
    return result

def nearest_neighbor_classifier(testx, trainx, testy, trainy, distance_function):
    test_length = len(testx)
    train_length = len(trainx)
    result = create_result_vector(test_length)
    for i in range(test_length):
        min_distance = distance_function(testx[i], trainx[0])
        label = trainy[0]
        for j in range(train_length):
            distance_j = distance_function(testx[i], trainx[j])
            if distance_j < min_distance:
                min_distance = distance_j
                label = trainy[j]
        result[i] = label
    return result

def calculate_err_rate(result, testy):
    err = 0
    for i in range(len(testy)):
        if result[i] != testy[i]:
            err += 1
    return float(err)/float(len(testy))

def confusion_matrix(result, testy):
    matrix = np.zeros(shape=(3,3))
    for i in range(len(result)):
            if testy[i] == 0 and result[i] == 0:
                matrix[0][0] += 1
            if testy[i] == 0 and result[i] == 1:
                matrix[0][1] += 1
            if testy[i] == 0 and result[i] == 2:
                matrix[0][2] += 1
            if testy[i] == 1 and result[i] == 0:
                matrix[1][0] += 1
            if testy[i] == 1 and result[i] == 1:
                matrix[1][1] += 1
            if testy[i] == 1 and result[i] == 2:
                matrix[1][2] += 1
            if testy[i] == 2 and result[i] == 0:
                matrix[2][0] += 1
            if testy[i] == 2 and result[i] == 1:
                matrix[2][1] += 1
            if testy[i] == 2 and result[i] == 2:
                matrix[2][2] += 1
    return matrix
                

result_l1 = nearest_neighbor_classifier(testx, trainx, testy, trainy, L1_distance)
err_rate_l1 = calculate_err_rate(result_l1, testy)
print "error rate for l1 distance: {0}".format(err_rate_l1)
print confusion_matrix(result_l1, testy)

result_l2 = nearest_neighbor_classifier(testx, trainx, testy, trainy, L2_distance)
err_rate_l2 = calculate_err_rate(result_l2, testy)
print "error rate for l2 distance: {0}".format(err_rate_l2)
print confusion_matrix(result_l2, testy)