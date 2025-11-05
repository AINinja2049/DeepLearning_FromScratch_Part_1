from typing import List
import numpy as np
from numpy import ndarray

class layer:
    pass

class loss:
    pass

# Main blueprint of making my neural network
class NeuralNetwork:
    def __init__(self,
                 layers: List[layer],
                 loss: loss,
                 learning_rate: float = 0.01) -> None:
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        print("My practice neural network initialized!")

nn = NeuralNetwork(layers=[layer(),layer()], loss=loss())

# Lets understand the simplest class and constructor __init__ first.
# We can define only constructor __init__ only once.

class Computers:
    def __init__(self,
                 company,
                 countryoforigin,
                 boughtby):
        self.company = company
        self.countryoforigin = countryoforigin
        self.boughtby = boughtby

Object = Computers("Hewlett-Packard","USA-Standford two friends flipped a \
                       coin", "Still owened by HP")
print("Pure Object",Object)
print("Instance of the object",Object.company)


#----Practice-2----#
class NeuralNN:
    def __init__(self,
                 layers:List[layer],
                 loss:loss,
                 learning_rate:float=0.01) -> None:
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        print("Neural network practice sesssion - 2")

NeuralNN = NeuralNN(layers=[layer(),layer()], loss=loss())

#-----Squares and LeakyRelu function-----#
def squares(x1: ndarray, x2: ndarray) -> ndarray:
    return pow(x1,2)+pow(x2,3)
print("Squares + Cubes = ",squares(2,2))

def leakyrelu(x: ndarray) -> ndarray:
    return np.maximum(0.2*x,x)

print("Leaky Relu: ",leakyrelu(np.array([-4,-3,-2,-1,0,1,2,3,4])))

#-----Function-----#
def f(input_: ndarray)-> ndarray:
    output = 2
    return output
print("Function general: ",f(np.array(2)))

def add(a: int, b: int) -> int:
    return a + b

print("addition check",add(4,3.3))

#-----Derivatives using callable-----#
from typing import Callable # Callable is type hint function it tells  

def deriv(func: Callable[[ndarray],ndarray], # Here it means a input is an array and output is also an array 
          input_: ndarray,
          delta: float = 0.001)-> ndarray:
    return (func(input_+delta)-func(input_-delta))/(2*delta)

def unknown(x: ndarray) -> ndarray:
    return np.power(x,3)

y = np.array([1,2,3,4,5])
print("Derivatives", deriv(unknown,y,delta=0.0001) )



#-----Nested functions-----#
from typing import List
from typing import Callable
import numpy as np
from numpy import ndarray


# A function that takes an ndarray as an argument and ndarray as output  
ArrayFunction = Callable[[ndarray],ndarray]

# Making a chain: A list of function f2(f2(x))
Chain = List[ArrayFunction]

# Let's define the function

def chain_length_2(chain: Chain,
                   x: ndarray) -> ndarray:
    '''Evaluates two nested functions
    '''
    assert len(chain)==2, "Length of the chain function should be 2"
    f1 = chain[0] #first list element is first function
    f2 = chain[1]
    return f2(f1(x))

# Example functions
def square(x: ndarray) -> ndarray:
    return np.power(x, 2)

def double(x: ndarray) -> ndarray:
    return 2 * x

twofunction = [square,double]

z = np.array([1,2,3,4,5])
y = chain_length_2(twofunction,z)
print("Chain function results", y)