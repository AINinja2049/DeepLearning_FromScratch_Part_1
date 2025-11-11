import numpy as np
from numpy import ndarray
from typing import Dict, Tuple


def forward_regression(X_batch: ndarray,
                       y_batch: ndarray,
                       weights: Dict[str,ndarray])->Tuple[float,Dict[str,ndarray]]:
    assert X_batch.shape[0]==y_batch.shape[0], "Checking X_batch row == y_batch rows"
    assert X_batch.shape[0]==weights['W'].shape[1],"Checking whether X.W is possible"
    N = np.dot(X_batch,weights['W'])
    P = N + weights['B']
    loss = np.mean(np.power(P-y_batch,2))
    forward_info: Dict[str,ndarray]={}
    forward_info['X'] = X_batch
    forward_info['N'] = N # Value of X.W
    forward_info['P'] = P # Value of X.W + b
    forward_info['L'] = loss
    return loss, forward_info