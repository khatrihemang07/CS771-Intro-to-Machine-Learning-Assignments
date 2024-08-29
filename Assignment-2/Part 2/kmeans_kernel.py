import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('kmeans_data.txt',delimiter='  ')

#Computing the Euclidean distance between data points.
def l2_norm(x,y):
    return np.sum((x[:, np.newaxis, :] - y[np.newaxis, :, :])**2, axis=2)

#Predicting the cluster of the input points.
def predict(x,y):
    return np.argmin(l2_norm(x,y), axis=1).reshape(-1, 1)
    
#Calcuating the mean of clusters.
def mean(x,y):
    zero_mat = np.vstack([np.mean(x[y == i], axis=0) for i in [0, 1]])
    return zero_mat

#Iteratively finding the cluster means.
for i in range(10):
    rand = np.random.randint(350,size=1).reshape(())
    train = np.exp(-0.1*np.sum(np.square(data - data[rand,:].reshape((1,-1))), axis=1)).reshape(-1,1)
    rev=train[:2,:]
    model = predict(train,rev)

    mn=mean(train,model)
    model = predict(train,mn)
    p=(model==1).reshape(model.shape[0])
    n=(model==0).reshape(model.shape[0])

    #Plotting the data in scatterplot 
    plt.figure(i)
    plt.scatter(data[p,0],data[p,1],c='g')
    plt.scatter(data[n,0],data[n,1],c='r')
    plt.plot(data[rand,0],data[rand,1],'b*')

plt.show()

