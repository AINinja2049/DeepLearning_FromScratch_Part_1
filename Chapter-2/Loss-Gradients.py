import numpy as np
from numpy import ndarray
from typing import Dict, List, Tuple

def forward_regression(X_batch: ndarray,
                       y_batch: ndarray,
                       weights: Dict[str,ndarray])->Tuple[float,Dict[str,ndarray]]:
    assert X_batch.shape[0]==y_batch.shape[0], "No. rows of X_batch == y_batch"
    assert X_batch.shape[0]==weights['W'].shape[1], "Shapes of multiplication must conform"
    assert weights['B'].shape[0]==weights['B'].shape[1], "Scalar value must only be assigned to bias, b"
    N = np.dot(X_batch,weights['W'])
    P = N + weights['B']
    loss = np.mean(np.power(P-y_batch,2))
# Storing
    forward_info: Dict[str,ndarray]={}
    forward_info['X'] = X_batch
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['L'] = loss
    return loss, forward_info

def loss_gradients(forward_info: Dict[str,ndarray],
                   weights: Dict[str, ndarray])->Dict[str,ndarray]:
    '''
    Compute dL/dW and dL/dB for step by step linear regression model.'''
    batch_size = forward_info['X'].shape[0]
    dLdP = -2*(forward_info['y']-forward_info['P'])
    dPdN = np.ones_like(forward_info['N'])
    dPdB = np.ones_like(forward_info['B'])
    dLdN = dLdP * dPdN
    dNdW = np.transpose(forward_info['X'],(1,0))
    dLdW = np.dot(dNdW,dLdN) # X^T on the left. Note on first chapter last section
    dLdB = (dLdP * dPdB).sum(axis = 0)
    loss_gradients: Dict[str, ndarray]={}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB
    return loss_gradients