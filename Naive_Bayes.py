import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
	
	## Outputs ##
	# log_product - float
    log_product = 0
    for i in x:
        log_product += i
    return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters alpha and beta, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, alpha, beta):
	## Inputs ## 
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
    	
	## Outputs ##
	# D - (2 by V) numpy ndarray
    D = np.zeros([2, XTrain.shape[1]])
    article_num = len(XTrain)
    word_list_len = len(XTrain[0])
    num_of_article_zero = 0
    num_of_article_one = 0
    for i in xrange(article_num):
        if yTrain[i] == 0:
            num_of_article_zero += 1
        else:
            num_of_article_one += 1
    count_1 = 0
    count_2 = 0

    for j in xrange(word_list_len):
        for i in xrange(article_num):
            if yTrain[i] == 0 and XTrain[i][j] == 1:
                count_1 += 1
            if yTrain[i] == 1 and XTrain[i][j] == 1:
                count_2 += 1
        
        a_1_0 = count_1
        a_0_0 = num_of_article_zero - a_1_0
        D[0][j] = ((a_1_0 + beta - 1) * 1.0) / ((a_1_0 + beta - 1) + (a_0_0 + alpha - 1))
        
        a_1_1 = count_2
        a_0_1 = num_of_article_one - a_1_1
        D[1][j] = ((a_1_1 + beta - 1) * 1.0) / ((a_1_1 + beta - 1) + (a_0_1 + alpha - 1))
        count_1 = 0
        count_2 = 0

    return D
	
# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float
    count = 0
    for i in yTrain:
        if i == 0:
            count += 1

    p = (count * 1.0) / len(yTrain)
    return p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
	
	## Outputs ##
	# yHat - 1D numpy ndarray of length m

    yHat = np.ones(XTest.shape[0])
    m = len(XTest)
    V = len(XTest[0])
    p_0 = 0
    p_1 = 0
    for x in xrange(m):
    	p_0 = math.log(p)
        p_1 = math.log(1 - p)
        for i in xrange(V):
            if XTest[x][i] == 1:
                p_0 += math.log(D[0][i])
                p_1 += math.log(D[1][i])
            else:
                p_0 += math.log(1 - D[0][i])
                p_1 += math.log(1 - D[1][i])

        if p_0 >= p_1:
            yHat[x] = 0
        else:
            yHat[x] = 1

    return yHat


# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float
    length = len(yHat)
    count = 0
    for i in xrange(length):
        if yHat[i] != yTruth[i]:
            count += 1
    error = (count * 1.0) / length
    return error


