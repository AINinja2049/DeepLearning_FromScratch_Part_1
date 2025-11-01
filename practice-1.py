from typing import List

class layer:
    pass

class loss:
    pass

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
        