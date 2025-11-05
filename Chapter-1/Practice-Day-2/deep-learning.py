from typing import List

class Layer:
    pass

class Loss:
    pass

class NeuralNetwork:
    def __init__(self,
                 layers: List[Layer],
                 loss: Loss,
                 learning_rate: float = 0.01) -> None:
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        print("Neural network initialized!")
    
nn = NeuralNetwork(layers=[Layer(), Layer()], loss=Loss())