# Importing numpy to handle files and perform mathematical operations.
import numpy as np

# Importing data given in the question and using meta-data from the question.

# (40 x N_i x D): 40 feature matrices. X_seen[i] is the N_i x D feature matrix of seen class i
x_seen = np.load('X_seen.npy',
                 encoding="bytes", allow_pickle=True)
# (6180, 4096): feature matrix of the test data.
x_test = np.load(
    'Xtest.npy')
# (6180, 1): ground truth labels of the test data
y_test = np.load(
    'Ytest.npy')
# (40, 85): 40x85 matrix with each row being the 85-dimensional class attribute vector of a seen class.
seen_attr = np.load(
    'class_attributes_seen.npy')
# (10, 85): 10x85 matrix with each row being the 85-dimensional class attribute vector of an  unseen class.
unseen_attr = np.load(
    'class_attributes_unseen.npy')

seen_attr = np.array(seen_attr)
unseen_attr = np.array(unseen_attr)

# Computing the mean of all seen classes.

mean_seen_classes = np.zeros((x_seen.shape[0], x_seen[0].shape[1]))
for i in range(0, x_seen.shape[0]):
    mean_seen_classes[i] = np.mean(
        x_seen[i], axis=0).reshape(1, x_seen[0].shape[1])

# Defining an array to store output of the model.

rows, cols = (40, 4096)
arr = np.zeros((rows, cols), dtype=int)
identity_mat = np.eye(85)

"""
Main function that performs prediction using the above data to predict label of test data and compare it with 
actaul label of test data and compute overall accuracy of the model.

"""

def predict(weight):
    mean = []
    mean = [np.dot(weight, i) for i in unseen_attr[:10]]
    mean_unseen_seen = np.array(mean)

    count = 0
    for i in range(6180):
        difference = []
        for j in range(10):
            squared_diff = (x_test[i] - mean_unseen_seen[j])**2
            sum_squared_diff = np.sum(squared_diff)
            dist = np.sqrt(sum_squared_diff)
            difference.append(dist)
        predicted_value = np.argmin(difference)+1
        if predicted_value == y_test[i]:
            count += 1
    accuracy = count/6180
    return accuracy


mat1 = np.dot(seen_attr.T, seen_attr)

mat2 = np.dot(np.transpose(seen_attr), mean_seen_classes)

for i in [0.01, 0.1, 1, 10, 20, 50, 100]:
    lam_identity = mat1 + i*identity_mat
    inverse = np.linalg.inv(lam_identity)
    w = np.dot(inverse, mat2)
    weight = np.transpose(w)
    accuracy = predict(weight)
    print(f"lambda={i} is ", accuracy*100)
