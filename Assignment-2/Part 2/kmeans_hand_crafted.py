import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('kmeans_data.txt',delimiter='  ')

#Calculating the Euclidean distance between two inputs.
def l2_norm(x,y):
    return np.sum((x[:, np.newaxis, :] - y)**2, axis=2)

#Predicting the cluster of inputs
def predict(x,y):
    return np.argmin(l2_norm(x,y), axis=1).reshape(-1, 1)
    
#Calculating the mean of cluster
def mean(x,y):
    zero_mat = np.vstack([np.mean(x[y == i], axis=0) for i in [0, 1]])
    return zero_mat

#Calculating the maximum absolute values of the dataset
train = np.max(np.abs(data), axis=1).reshape(-1, 1)
centres=train[:2,:]
model= predict(train,centres)

#Plotting the raw data on a scatter plot.
# Scatter plot
plt.scatter(data[:, 0], data[:, 1], color='black')  # 'viridis' is just an example colormap

# Adding labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot of Data with Color-coded Train Values')

plt.show()

#Computing the mean iteratively to assign clusters appropriately.
for i in range(10):
    centres=mean(train,model)
    model=predict(train,centres)
    p=(model==1).reshape(model.shape[0])
    n=(model==0).reshape(model.shape[0])

    #Plotting the separated data on the scatterplot.
    plt.scatter(data[p,0], data[p,1], c='g')
    plt.scatter(data[n,0], data[n,1], c='r')

plt.show()

