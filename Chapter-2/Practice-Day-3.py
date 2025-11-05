import numpy as np
from numpy import ndarray
from typing import Dict, Tuple

def forward_regression(X_batch: ndarray,
                       y_batch: ndarray,
                       weights: Dict[str,ndarray])->Tuple[float,Dict[str,ndarray]]:
    assert X_batch.shape[0]==y_batch.shape[0], "Checking if the features batch rows are equal to target rows"
    assert X_batch.shape[1]==weights['W'].shape[1], "Checking whether X.W is possible"
    assert weights['B'].shape[0]==weights['B'].shape[1]==1, "Making sure the bias is only 1 single scalar value spread across"
    N = np.dot(X_batch,weights['W'])
    P = N + weights['B']
    loss = np.mean(np.power(P-y_batch,2))
    forward_info: Dict[str, ndarray]={}
    forward_info['X'] = X_batch
    forward_info['N'] = N # Value after x1*w1 + x2*w2 + x2*w3 + ...
    forward_info['P'] = P # Value after x1*w1 + x2*w2 + x2*w3 + ... + b
    forward_info['L'] = loss # Value of mean square error
    return loss, forward_info 
### This the code for linear regression forward pass

