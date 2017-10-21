import os
import math
import numpy as np


def LogReg_ReadInputs(filepath):
    
    #function that reads all four of the Logistic Regression csv files and outputs
    #them as such

    #Input
    #filepath : The path where all the four csv files are stored.

    #output 
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features


    XTrain = np.genfromtxt(os.path.join(filepath, 'LogReg_XTrain.csv'), delimiter=',')
    #XTrain : NxK numpy matrix containing N number of K+1 dimensional training features

    yTrain = np.genfromtxt(os.path.join(filepath, 'LogReg_yTrain.csv'), delimiter=',')
    #yTrain : Nx1 numpy vector containing the actual output for the training features

    XTest = np.genfromtxt(os.path.join(filepath, 'LogReg_XTest.csv'), delimiter=',')
    #XTest  : nxK numpy matrix containing n number of K+1 dimensional testing features

    yTest = np.genfromtxt(os.path.join(filepath, 'LogReg_yTest.csv'), delimiter=',')
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    i_XTrain = len(XTrain)
    i_XTest = len(XTest)

    One_1 = np.ones((i_XTrain,1))
    One_2 = np.ones((i_XTest,1))
    XTrain = np.column_stack([One_1, XTrain])
    XTest = np.column_stack([One_2, XTest])

    return (XTrain, yTrain, XTest, yTest)
    
def LogReg_CalcObj(X, y, w):
    
    #function that outputs the conditional log likelihood we want to maximize.

    #Input
    #w      : numpy weight vector of appropriate dimensions initialized to 0.5
    #AND EITHER
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #OR
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    #Output
    #cll   : The conditional log likelihood we want to maximize
    N = len(X)
    cll = 0
    for i in xrange(N):
        p_i = 1.0/(1+math.e**(-np.dot(X[i],w)))
        if y[i] == 1: 
            cll += math.log(p_i)
        if y[i] == 0:
            cll += math.log(1 - p_i)

    cll *=  1/(N * 1.0)
    
    return cll
    
def LogReg_CalcSG(x, y, w):
    
    #Function that calculates and returns the stochastic gradient value using a
    #particular data point (x, y).

    #Input
    #x : 1x(K+1) dimensional feature point
    #y : Actual output for the input x
    #w : weight vector 

    #Output
    #sg : gradient of the weight vector
    p_i = 1.0/(1 + math.e ** (-np.dot(x, w)))
    sg = (y - p_i) * x
    return sg

        
def LogReg_UpdateParams(w, sg, eta):
    
    #Function which takes in your weight vector w, the stochastic gradient
    #value sg and a learning constant eta and returns an updated weight vector w.

    #Input
    #w  : weight vector before update 
    #sg : gradient of the calculated weight vector using stochastic gradient ascent
    #eta: Learning rate

    #Output
    #w  : Updated weight vector
    w = w + eta * sg

    return w
    
def LogReg_PredictLabels(X, y, w):
    
    #Function that returns the value of the predicted y along with the number of
    #errors between your predictions and the true yTest values

    #Input
    #w : weight vector 
    #AND EITHER
    #XTest : nx(K+1) numpy matrix containing m number of d dimensional testing features
    #yTest : nx1 numpy vector containing the actual output for the testing features
    #OR
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    
    #Output
    #yPred : An nx1 vector of the predicted labels for yTest/yTrain
    #perMiscl : The percentage of y's misclassified
    length = len(X)
    yPred = []
    for i in xrange(length):
        p_i = 1.0/(1 + math.e ** (-np.dot(X[i],w)))
        if p_i > 0.5:
            yPred.append(1)
        else:
            yPred.append(0)
    error_num = 0

    for i in xrange(length):
        if yPred[i] != y[i]:
            error_num += 1
    perMiscl = (error_num * 1.0) / length
    return (yPred, perMiscl)    


def LogReg_SGA(XTrain, yTrain, XTest, yTest):
    
    #Stochastic Gradient Ascent Algorithm function

    #Input
    #XTrain : Nx(K+1) numpy matrix containing N number of K+1 dimensional training features
    #yTrain : Nx1 numpy vector containing the actual output for the training features
    #XTest  : nx(K+1) numpy matrix containing n number of K+1 dimensional testing features
    #yTest  : nx1 numpy vector containing the actual output for the testing features

    #Output
    #w             : final weight vector
    #trainPerMiscl : a vector of percentages of misclassifications on your training data at every 200 gradient descent iterations
    #testPerMiscl  : a vector of percentages of misclassifications on your testing data at every 200 gradient descent iterations
    #yPred         : a vector of your predictions for yTest using your final w
    


    K_plus_one = len(XTrain[0])
    N = len(XTrain)
    n = len(XTest)
    loop = 200
    y_te = 0
    w = np.ones((K_plus_one), dtype = float) * 0.5
    
    trainPerMiscl = []
    testPerMiscl = []
    loop_i = 0
    for m in xrange(5):
        for i in xrange(N):
            loop_i += 1
            sg = LogReg_CalcSG(XTrain[i], yTrain[i], w)
            w = LogReg_UpdateParams(w, sg, 0.5/math.sqrt(loop_i))
            if loop_i % 200 == 0:
                (y_tr, p_tr) = LogReg_PredictLabels(XTrain, yTrain, w)
                (y_te, p_te) = LogReg_PredictLabels(XTest, yTest, w)
                trainPerMiscl.append(p_tr)
                testPerMiscl.append(p_te)
        
    yPred = y_te
    return (w, trainPerMiscl, testPerMiscl, yPred)

#(XTrain, yTrain, XTest, yTest) = LogReg_ReadInputs("../data/")

#print(LogReg_SGA(XTrain, yTrain, XTest, yTest))
def plot():     # This function's results should be returned via gradescope and will not be evaluated in autolab.
    
    return None