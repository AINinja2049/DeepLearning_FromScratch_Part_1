import numpy as np
from numpy import ndarray
from typing import List
from typing import Callable

# Define a function 
ArrayFunction = Callable[[ndarray],ndarray]

# Define a chain as List of function
Chain = List[ArrayFunction]

# Define a forward pass for multiple row features and multiple column neurons
# (X,W) -> X.W -> N -> S(N)
def sigmoid(x: ndarray)-> ndarray:
    return 1/(1+np.exp(-x))
# 
def forward_pass(X: ndarray,
                 W: ndarray,
                 sigma: Callable[[ndarray],ndarray])->ndarray:
    
    assert X.shape[1]==W.shape[0], "No. of features (X rows) = No. of layers of neurons (W cols)"
    N = np.dot(X,W)
    S = sigma(N)
    L = np.sum(S)
    return L
# Derivative function for use
def deriv(func: Callable[[ndarray],ndarray],
          input_: ndarray,
          delta: float=0.001)-> ndarray:
    return (func(input_+delta)-func(input_-delta))/(2*delta)

# Define backprop for 2D, 3 feature vectors and 2 neuron layers
def backprop(X: ndarray,
             W: ndarray,
             sigma: Callable[[ndarray],ndarray])->ndarray:
    assert X.shape[1]==W.shape[0], "X rows == W cols?"
    N = np.dot(X,W)
    # dL/dS * dS/dN * dN/dX
    S = sigma(N)
    L = np.sum(S)
    dLdS = np.ones_like(S)       # This is because loss functio is L = S1 + S2 + S3 + ... 
    dSdN = deriv(sigma,N)
    dLdN = dLdS*dSdN
    dNdX = np.transpose(W,(1,0)) # N = (x1*w1+x2*w2+x3*w3) = N'(X) = [w1 w2 w3] = W^T
    dNdW = np.transpose(X,(1,0)) # N = (x1*w1+x2*w2+x3*w3) = N'(W) = [x1 x2 x3]^T
    
    dL_dX = np.dot(dLdN,dNdX)
    dL_dW = np.dot(dNdW,dLdN)
    total_gradient = dL_dX + dL_dW
    return total_gradient