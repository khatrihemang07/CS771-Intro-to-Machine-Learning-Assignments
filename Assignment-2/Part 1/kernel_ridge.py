import numpy as np
import matplotlib.pyplot as plt

train = np.genfromtxt('ridgetrain.txt',delimiter='  ')
test = np.genfromtxt('ridgetest.txt',delimiter='  ')

#Extracting data from the datasets
x_train, y_train, x_test, y_test = train[:,0], train[:,1], test[:,0], test[:,1]

kernel = np.exp((-0.1)*np.square(x_train.reshape((-1,1))-x_train.reshape((1,-1))))

#Identity matrix 
identity = np.eye(x_train.shape[0])

for values in [0.1,1,10,100]:
    kernel_inv=np.linalg.inv(kernel+values*identity)
    alpha = np.dot(kernel_inv,y_train.reshape((-1,1)))
    
    #Applying Radial Basis Function(RBF) kernel
    kernel_test=np.exp((-0.1)*np.square(x_train.reshape((-1,1))-x_test.reshape((1,-1))))
    y_pred=np.dot(alpha.T,kernel_test).reshape((-1,1))

    #Calculating Root Mean Square Error(RMSE)
    rmse=np.sqrt(np.mean(np.square(y_test.reshape((-1,1))-y_pred)))
    print ('RMSE for lambda = ' + str(values) + ' is ' + str(rmse))

    #Plotting vaues in scatterplot using matplotlib library
    plt.figure(values)
    plt.title('lambda = ' + str(values) + ', rmse = ' + str(rmse))
    plt.plot(x_test, y_pred, 'r*')
    plt.plot(x_test, y_test, 'b*')
    
plt.show()
