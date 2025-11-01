import numpy as np
from numpy import ndarray
from typing import Callable
from typing import List
import matplotlib.pyplot as plt
########## Lets define derviative

def deriv(func: Callable[[ndarray],ndarray],
          input_: ndarray,
          delta: float = 0.001)->ndarray:
    return (func(input_+delta)-func(input_-delta))/(2*delta)

########## Lets define chained function
ArrayFunction = Callable[[ndarray],ndarray]
########## Lets define the list of ArrayFunction
Chain = List[ArrayFunction]

########## Lets define the chain function. It will be a general function 
def chain_length_2(chain: Chain,
                  x: ndarray)->ndarray:
    assert len(chain)==2, "Input function list must be of two function inside the list"
    f1 = chain[0]
    f2 = chain[1]
    return f2(f1(x))

########## Lets define different functions
def square(x: ndarray)->ndarray:
    return np.power(x,2)

def cubic(x: ndarray)->ndarray:
    return np.power(2*x,3)

def sigmoid(x: ndarray)->ndarray:
    return 1/(1+np.exp(-x))

def leakyrelu(x: ndarray)->ndarray:
    return np.maximum(0.2*x,x)
Functions = [square,cubic]
values = np.array([0,1,2,3,4.555,6.2253325])
ChainedFunction = chain_length_2(Functions,values)
print("Evaluated Chained Function of Length 2 is: ", ChainedFunction)

########## Lets define chain derivative now that central differentiation is done
########## 2 nested functions
def chain_deriv_2(chain: Chain,
                  input_range: ndarray)-> ndarray:
    assert len(chain)==2, "Input function must be of length 2"
    assert input_range.ndim==1, "Functions requires 1 dimensional arrays"
    f1 = chain[0]
    f2 = chain[1]

    f1_of_x = f1(input_range)
    df1_of_x = deriv(f1,input_range)
    df2_of_f1 = deriv(f2,f1_of_x) 
    return df2_of_f1*df1_of_x

chain1 = [square, sigmoid]
chain2 = [sigmoid, square]

investigation_range = np.arange(-3,3,0.01)
print(investigation_range.ndim)
plt.plot(investigation_range, chain_length_2(chain1,investigation_range))
plt.plot(investigation_range, chain_deriv_2(chain1,investigation_range))
plt.show()


plt.plot(investigation_range, chain_length_2(chain2,investigation_range))
plt.plot(investigation_range, chain_deriv_2(chain2,investigation_range))
plt.show()

########## Derivative for 3 nested functions
def chain_deriv_3(chain: Chain,
                  input_range: ndarray)-> ndarray:
    assert len(chain)==3, "Three nested functions or not?"
    assert input_range.ndim==1, "1-D array of domain or not?"
    f1 = chain[0]
    f2 = chain[1]
    f3 = chain[2]

    # f1(x)
    f1_x = f1(input_range)
    # f2(f1(x))
    f2_f1_x = f2(f1_x)

    # f1'(x)
    df1_x = deriv(f1, input_range)
    # f2'(f1(x))
    df2_f1_x = deriv(f2, f1_x)
    # f3'(f2(x)) 
    df3_f2_f1_x = deriv(f2,f2_f1_x)
    
    # total derivative = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
    return df1_x*df2_f1_x*df3_f2_f1_x

chain3 = [leakyrelu,sigmoid,square ]
#plt.plot(investigation_range, chain_length_3(chain3,investigation_range))
plt.plot(investigation_range, chain_deriv_3(chain3,investigation_range))
plt.show()
