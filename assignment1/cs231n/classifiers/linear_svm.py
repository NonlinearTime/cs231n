# This Python file uses the following encoding: utf-8
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  dw = []

    # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  count = 0
  for i in xrange(num_train):
    count = 0
    dw = []
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
        if j == y[i]:
            #dw[j] += - margin*x[i].T
            continue
        margin = scores[j] - correct_class_score + 1 # note delta = 1
        if margin > 0:
            count +=1
            #dW[:,j] += X[i]
            #print(1)
            dw.append(X[i])
            loss += margin
        else:
            #print(1)
            dw.append(np.zeros(W.shape[0]))
    #dW[:,y[i]] += - count*X[i]
    dw.insert(y[i],(- count*X[i]))
    #print(np.array(dw).T.shape,dW.shape)
    #print(dW.shape)
    try:
        dW = dW + np.array(dw).T
    except:
        print(np.array(dw).T.shape,np.array(dw).T.size)
        
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #dW = dW + 2 * reg *dW

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  raw = X.dot(W)
  loc = [tuple(range(y.size)),tuple(y.reshape(y.size).tolist())]
  #print(loc)
  subitem = raw[loc][:,np.newaxis]
  pre = raw - subitem
  res = pre + 1
  res = (res + np.abs(res))/2
  res[loc] = 0
  loss = np.sum(res)/num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  res[res > 0] = 1.0                         # 示性函数的意义
  row_sum = np.sum(res, axis=1)                  # 1 by N
  res[loc] = -row_sum        
  dW += np.dot(X.T, res)/num_train #+ reg * W     # D by C
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
