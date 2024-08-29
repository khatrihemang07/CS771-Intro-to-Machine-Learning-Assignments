#Importing libraries to handle files and visualize
import numpy as np
import matplotlib.pyplot as plt 

#Importing data from txt file using genfromtxt function and separating using (" ") space.
train = np.genfromtxt('ridgetrain.txt',delimiter='  ')
test = np.genfromtxt('ridgetest.txt',delimiter='  ')

#Splitting data using slicing
x_train, y_train, x_test, y_test = train[:,0], train[:,1], test[:,0], test[:,1]

#Iterating over values given in the question.
for values in [2,5,20,50,100]:
    #Selecting random inputs using random choice function with no repetition
    landmarks = np.random.choice(x_train,values,replace=False)

    #Creating identity matrix of size value
    identity = np.eye(values)

    #Using RBF kernel as desired in the question
    final_train = np.exp(-0.1*np.square(x_train.reshape((-1,1)) - landmarks.reshape((1,-1))))
    final_train_dot = np.dot(final_train.T,final_train)

    #Calculating weight to be used in the prediction process.
    W=np.dot(np.linalg.inv(final_train_dot+0.1*identity),np.dot(final_train.T,y_train.reshape((-1,1))))

    final_test=np.exp(-0.1*np.square(x_test.reshape((-1,1)) - landmarks.reshape((1,-1))))
    y_pred = np.dot(final_test,W)

    #Calculating Root Mean Square Error(RMSE)
    rmse = np.sqrt(np.mean(np.square(y_test.reshape((-1,1)) - y_pred)))
    print ('RMSE for lambda = ' + str(values) + ' is ' + str(rmse))

    #Plotting the data on graph using scatterplot
    plt.figure(values)
    plt.title('Landmark values = ' + str(values) + ', rmse = ' + str(rmse))
    plt.plot(x_test, y_pred, 'r*')
    plt.plot(x_test, y_test, 'b*')

plt.show()