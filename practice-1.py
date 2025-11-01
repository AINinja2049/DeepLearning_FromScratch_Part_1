from typing import List

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
        