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