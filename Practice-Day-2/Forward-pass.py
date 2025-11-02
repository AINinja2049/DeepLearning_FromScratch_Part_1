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
    
    assert X.shape[1]==W.shape[0], "No. of features (X rows) = No. of neurons (W cols)"
    N = np.dot(X,W)
    S = sigma(N)
    L = np.sum(S)
    return L