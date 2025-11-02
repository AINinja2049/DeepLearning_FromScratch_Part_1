import numpy as np
from numpy import ndarray
from typing import List
from typing import Callable


########## Array function
ArrayFunction = Callable[[ndarray],ndarray]
########## Chain function
Chain = List[ArrayFunction]

def multiple_inputs_add(x: ndarray,
                        y: ndarray,
                        sigma: ArrayFunction)->float:
    '''Function with multiple input and addition and forward pass
    '''
    assert x.shape==y.shape, "Are the two variable's shape equal?"
    a = x+y
    return sigma(x+y)
########## Derivatives of multiple functions
def deriv(func: Callable[[ndarray],ndarray],
          input_: ndarray,
          delta: float = 0.001)->ndarray:
    return (func(input_+delta)-func(input_-delta))/(2*delta) 

def multiple_inputs_add_backward(x: ndarray,
                                 y: ndarray,
                                 sigma: ArrayFunction)->float:
    a = x+y
    dsda = deriv(sigma,a)
    dadx,dady = 1,1
    return  dsda*dadx*dady

########## Matrix multiplication vectors: Functions with multiple vector inputs
def matmulforward(X: ndarray,
                  W: ndarray)->ndarray:
    assert X.shape[1]==W.shape[0], format(X.shape[1],W.shape[0])
    '''For matrix multiplication, the number of columns in the first array should
    match the number of rows in the second; instead the number of columns in the
    first array is {0} and the number of rows in the second array is {1}. X = row, W = col
    '''
    N = np.dot(X,W)
    return N

########## Derivatives of Functions with multiple vector inputs
def matmul_backward_first(X:ndarray,
                   W:ndarray)->ndarray:
    dNdX = np.transpose(W,(1,0))
    return dNdX
########## Vector functions and their derivatives: one step further
def matrixforwardextra(X: ndarray,
                       W: ndarray,
                       sigma: ArrayFunction)->ndarray:
    assert X.shape[1]==W.shape[0]
    N = np.dot(X,W)
    S = sigma(N)
    return S
########## Vector functions and their derivatives: the backward pass
def matrixfunction_backward_1(X: ndarray,
                              W: ndarray,
                              sigma: ArrayFunction)->ndarray:
    assert X.shape[1]==W.shape[0]
    N = np.dot(X,W) # X = row, W = col
    # v(X,W) = N = x1*w1 + x2*w2 + x3*w3
    # dv/dX = [w1 w2 w3] = W^T
    # Sigma(v(X,W)) = S(v(X,W))
    # dS/dX = S'(v(X,W))*v'(X,W)
    # dS/dX = dS/dN(N)*dN/dX(X,W)
    # dS/dX = 
    S = sigma(N)
    dSdN = deriv(sigma,N) # This will be a number
    dNdX = np.transpose(W,(1,0))
    # np.dot(number, array) = number * np.array
    # np.dot(array1, array2) = array1[0]*array2[0] + array1[1]*array2[1] + array1[2]*array2[2]
    return np.dot(dSdN,dNdX)

np.random.seed(190203)

X = np.random.randn(1,3)
W = np.random.randn(3,1)
def sigmoid(x: ndarray)->ndarray:
    return 1/(1+np.exp(-x))


print(X)
matmul_backward_first(X, W)
old = matrixfunction_backward_1(X,W,sigmoid)
print(old)
X1 = X.copy()
X1[0,2] +=0.01
new = matrixfunction_backward_1(X1,W,sigmoid)
print(new-old)
print((matrixforwardextra(X1,W,sigmoid)-matrixforwardextra(X,W,sigmoid))/0.01)


########## Computational graph with 2D matrix inputs
def matrix_function_forward(X: ndarray,
                            W: ndarray,
                            sigma: ArrayFunction)->ndarray:
    assert X.shape[1]==W.shape[0]
    N = np.dot(X,W)
    S = sigma(N)
    L = np.sum(S)
    return L
# L = G(S(N(X,W)))
# dL/dX = dG/dS * dS/dN * dN/dX

########## Backpropagation
def backpropagation(X: ndarray,
                    W: ndarray,
                    sigma: ArrayFunction)->ndarray:
    assert X.shape[1]==W.shape[0]
    N = np.dot(X,W)
    S = sigma(N)
    L = np.sum(S)
    dLdS = np.ones_like(S)
    dSdN = deriv(sigma,N)
    dLdN = dLdS*dSdN
    dNdX = np.transpose(W,(1,0))
    dLdX = np.dot(dLdN,dNdX)
    return dLdX

np.random.seed(190204)
X = np.random.randn(3,3)
W = np.random.randn(3,2) #2 is the number of neurons = col
print(X)
print(matrix_function_forward(X,W,sigmoid))
print(backpropagation(X,W,sigmoid))
X1 = X.copy()
X1[0,0] +=0.001
print((matrix_function_forward(X1,W,sigmoid) - matrix_function_forward(X,W,sigmoid))/0.001)