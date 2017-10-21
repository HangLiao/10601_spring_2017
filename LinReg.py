import os
import math
import numpy as np


    #Input
    #filepath : The path where all the four csv files are stored.

    #output 
    #XTrain : NxK+1 numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nxK+1 numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

def LinReg_ReadInputs(filepath):
    #with open(os.path.join(data_dir, 'vocabulary.csv'), 'rb') as f:
    #    reader = csv.reader(f)
    #    vocabulary = list(x[0] for x in reader)

    XTrain = np.genfromtxt(os.path.join(filepath, 'LinReg_XTrain.csv'), delimiter=',')
    #XTrain : NxK numpy matrix containing N number of K+1 dimensional training features

    yTrain = np.genfromtxt(os.path.join(filepath, 'LinReg_yTrain.csv'), delimiter=',')
    #yTrain : Nx1 numpy vector containing the actual output for the training features

    XTest = np.genfromtxt(os.path.join(filepath, 'LinReg_XTest.csv'), delimiter=',')
    #XTest  : nxK numpy matrix containing n number of K+1 dimensional testing features

    yTest = np.genfromtxt(os.path.join(filepath, 'LinReg_yTest.csv'), delimiter=',')
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    i_XTrain = len(XTrain)
    i_XTest = len(XTest)
    i_total = i_XTrain + i_XTest
    j_total = len(XTrain[0])

    for j in xrange(j_total):
        max_x = XTrain[0][j]
        min_x = XTrain[0][j]
        for i in xrange(i_total):
            if i < i_XTrain:
                if XTrain[i][j] < min_x:
                    min_x = XTrain[i][j]
                if XTrain[i][j] > max_x:
                    max_x = XTrain[i][j]
        for i in xrange(i_total):
            if i < i_XTrain:
                XTrain[i][j] = (XTrain[i][j] - min_x) / ((max_x - min_x) * 1.0)
            elif i >= i_XTrain:
                XTest[i - i_XTrain][j] = (XTest[i - i_XTrain][j] - min_x) / ((max_x - min_x) * 1.0)       

    One_1 = np.ones((i_XTrain,1))
    One_2 = np.ones((i_XTest,1))
    XTrain = np.column_stack([One_1, XTrain])
    XTest = np.column_stack([One_2, XTest])

    return (XTrain, yTrain, XTest, yTest)
    
def LinReg_CalcObj(X, y, w):
    
    #function that outputs the value of the loss function L(w) we want to minimize.

    #Input
    #w      : numpy weight vector of appropriate dimensions
    #AND EITHER
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #OR
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features
    
    X_times_w = np.dot(X,w)
    X_times_w_minus_y = X_times_w - y
    lossVal = 0.0
    n = len(X_times_w_minus_y)
    for i in range(n):
        lossVal += (X_times_w_minus_y[i] ** 2)
    lossVal *= (1/(n*1.0))
    #Output
    #loss   : The value of the loss function we want to minimize
    return lossVal


def LinReg_CalcSG(x, y, w):
    
    #Function that calculates and returns the stochastic gradient value using a
    #particular data point (x, y).
    
    #Input
    #x : 1x(K+1) dimensional feature point
    #y : Actual output for the input x
    #w : (K+1)x1 dimensional weight vector 
    
    #Output
    #sg : gradient of the weight vector
    product = np.dot(x, w)
    sg = -2 * (y - product) * np.transpose(x)
    return sg
    

def LinReg_UpdateParams(w, sg, eta):
    
    #Function which takes in your weight vector w, the stochastic gradient
    #value sg and a learning constant eta and returns an updated weight vector w.
    
    #Input
    #w  : (K+1)x1 dimensional weight vector before update 
    #sg : gradient of the calculated weight vector using stochastic gradient descent
    #eta: Learning rate

    #Output
    #w  : Updated weight vector
    w = w - eta * sg
    return w

def LinReg_SGD(XTrain, yTrain, XTest, yTest):
    
    #Stochastic Gradient Descent Algorithm function

    #Input
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional test features
    #yTest  : nx1 numpy vector containing the actual output for the test features
    
    #Output
    #w    : Updated Weight vector after completing the stochastic gradient descent
    #trainLoss : vector of training loss values at each epoch
    #testLoss : vector of test loss values at each epoch
    K_plus_one = len(XTrain[0])
    N = len(XTrain)
    n = len(XTest)
    loop = 100

    w = np.ones((K_plus_one), dtype = float) * 0.5
    
    trainLoss = []
    testLoss = []
    loop_num = 0

    for x in range(loop):
        for i in xrange(N):
            loop_num += 1
            sg = LinReg_CalcSG(XTrain[i], yTrain[i], w)
            
            w = LinReg_UpdateParams(w, sg, 0.5/math.sqrt(loop_num))

        trainLoss.append(LinReg_CalcObj(XTrain, yTrain, w))
        testLoss.append(LinReg_CalcObj(XTest, yTest, w))
    #w = np.ones((K_plus_one, 1), dtype = float) * 0.5
    

    return (w, trainLoss, testLoss)

#(XTrain, yTrain, XTest, yTest) = LinReg_ReadInputs("../data/")
#print(XTrain, XTest)
#print(LinReg_SGD(XTrain, yTrain, XTest, yTest))

def plot():     # This function's results should be returned via gradescope and will not be evaluated in autolab.
    
    return None
    

    
    