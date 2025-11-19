import numpy as np
from numpy import ndarray
from typing import List, Dict, Tuple

def sigmoid(x: ndarray)->ndarray:
    return 1/(1+np.exp(x))
    

def forward_loss(X: ndarray,
                 y: ndarray,
                 weights: Dict[str, ndarray]
                 ) -> Tuple[Dict[str, ndarray], float]:
    '''Compute the forward pass and the loss for the step
       by step neural network model.
    '''
    M1 = np.dot(X,weights['W1'])
    N1 = M1 + weights['B1']
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights['W2'])
    P  = M2 + weights['B2']
    loss = np.mean(np.power(y-P,2))

    forward_info: Dict[str,ndarray]={}
    forward_info['X']  = X
    forward_info['M1'] = M1
    forward_info['O1'] = O1
    forward_info['M2'] = M2
    forward_info['P']  = P
    forward_info['y']  = y

    return forward_info, loss

def loss_gradients(forward_info: Dict[str, ndarray],
                   weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
    '''Compute the partial derivatives of the loss
    with respect to each of the parameters in the neural network
    '''
    dLdP   = -2*(forward_info['y']-forward_info['P'])## ## ##
    dPdM2  = np.ones_like(forward_info['M2']) ##
    dLdM2  = dLdP * dPdM2 # ##
    dPdB2  = np.ones_like(forward_info['B2'])##
    dLdB2  = (dLdP * dPdB2).sum(axis=0) #
    
    dM2dW2 = np.transpose(forward_info['O1'],(1,0)) ##
    dLdW2  = np.dot(dM2dW2,dLdP) #

    dM2dO1 = np.transpose(['W2'],(1,0))
    dLdO1  = np.dot(dLdM2,dM2dO1)
    dO1dN1 = sigmoid(forward_info['N1'])*(1-forward_info['N1'])
    dLdN1  = dLdO1 * dO1dN1
    dN1dB1 = np.ones_like(weights['B1'])
    dN1dM1 = np.ones_like(forward_info['M1'])
    dLdB1  = (dLdN1 * dN1dB1).sum(axis=0)  

    dLdM1  = dLdN1 * dN1dM1
    dM1dW1 = np.transpose(forward_info['X'],(1,0))
    dLdW1  = np.dot(dM1dW1,dLdM1)

    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients['W2'] = dLdW2
    loss_gradients['B2'] = dLdB2
    loss_gradients['W1'] = dLdW1
    loss_gradients['B1'] = dLdB1
    return loss_gradients



    