import numpy as np
from numpy import ndarray
from typing import Dict, List, Tuple
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#### Preparation of dataset ####
diabetes = load_diabetes()
data = diabetes.data
target = diabetes.target
features = diabetes.feature_names


#### Slightly different from the book #### Always use the training mean and std to scale.
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=80718)
s = StandardScaler()
s.fit(X_train)
X_train = s.transform(X_train)
X_test = s.transform(X_test)
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)


#from sklearn.linear_model import LinearRegression
#lr = LinearRegression(fit_intercept=True)
#lr.fit(X_train, y_train)
#preds = lr.predict(X_test)
#plt.scatter(preds, y_test)
#plt.plot([0, 300], [0, 300])
#plt.show()


#### Weights and bias initializer ####
def init_weights(n_in: int)-> Dict[str,ndarray]:
    '''Initialize weights on first forward pass of model
    '''
    weights: Dict[str, ndarray] = {}
    W = np.random.randn(n_in,1)
    B = np.random.randn(1,1)
    weights['W'] = W
    weights['B'] = B
    return weights



#### Forward Loss ####
def forward_loss(X_batch: ndarray,
                       y_batch: ndarray,
                       weights: Dict[str,ndarray])->Tuple[float,Dict[str,ndarray]]:
    assert X_batch.shape[0]==y_batch.shape[0], "Checking if the features batch rows are equal to target rows"
    assert X_batch.shape[1]==weights['W'].shape[0], "Checking whether X.W is possible"
    assert weights['B'].shape[0]==weights['B'].shape[1]==1, "Making sure the bias is only 1 single scalar value spread across"
    N = np.dot(X_batch,weights['W'])
    P = N + weights['B']
    loss = np.mean(np.power(P-y_batch,2))
    forward_info: Dict[str, ndarray]={}
    forward_info['X'] = X_batch
    forward_info['N'] = N # Value after x1*w1 + x2*w2 + x2*w3 + ...
    forward_info['P'] = P # Value after x1*w1 + x2*w2 + x2*w3 + ... + b
    forward_info['y'] = y_batch
    forward_info['L'] = loss # Value of mean square error
    return loss, forward_info 




#### Helper ####
def to_2d_np(a: ndarray,
             type: str = "col")->ndarray:
    '''Turns a 1D Tensor into 2D
    '''
    assert a.ndim ==1, "Input tensors must be 1 dimensional"
    if type == "col":
        return a.reshape(-1,1)
    elif type == "row":
        return a.reshape(-1,1)



#### Helper permutations ####
def permute_data(X: ndarray,
                 y: ndarray):
    '''Permute X and y, using the same permutation, along axis=0
    '''
    perm = np.random.permutation(X.shape[0])
    return X[perm],y[perm]



Batch = Tuple[ndarray,ndarray]
def generate_batch(X: ndarray,
                   y: ndarray,
                   start: int = 0,
                   batch_size: int = 10)-> Batch:
    '''Generate batch from X and y, given a start position
    '''
    assert X.ndim == y.ndim == 2, "X and Y must be 2 dimensional"
    if start+batch_size > X.shape[0]:
        batch_size = X.shape[0]-start
    X_batch, y_batch = X[start:start+batch_size], y[start:start+batch_size]
    return X_batch, y_batch



#### Loss gradients ####
def loss_gradients(forward_info: Dict[str,ndarray],
                   weights: Dict[str, ndarray])->Dict[str,ndarray]:
    '''Compute dL/dW and dL/dB for step by step linear regression model.
    '''
    batch_size = forward_info['X'].shape[0]
    dLdP = -2*(forward_info['y']-forward_info['P'])
    dPdN = np.ones_like(forward_info['N'])
    dPdB = np.ones_like(weights['B'])
    dLdN = dLdP * dPdN
    dNdW = np.transpose(forward_info['X'],(1,0))
    dLdW = np.dot(dNdW,dLdN) # X^T on the left. Note on first chapter last section
    dLdB = (dLdP * dPdB).sum(axis = 0)
    loss_gradients: Dict[str, ndarray]={}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB
    return loss_gradients



#### Training Loop ####
def train(X: ndarray,
          y: ndarray,
          n_iter: int = 1000,
          learning_rate: float = 0.01,
          batch_size: int = 100,
          return_losses: bool = False,
          return_weights: bool = False,
          seed: int =1)-> None:
    if seed:
        np.random.seed(seed)
    start = 0
    # Initializing weights
    weights = init_weights(X.shape[1])
    # Permute data
    X,y = permute_data(X,y)
    if return_losses:
        losses= []

    for i in range(n_iter):
        # Generate batch
        if start >= X.shape[0]:
            X,y = permute_data(X,y)
            start = 0
        X_batch, y_batch = generate_batch(X,y, start, batch_size)
        start += batch_size
        # Train net using generated batch
        loss, forward_info = forward_loss(X_batch, y_batch, weights)
        if return_losses:
            losses.append(loss)
        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]
    if return_weights:
        return losses, weights
    
    return None


##### Training the linear regression #####
train_info = train(X_train, y_train,
                   learning_rate=0.001,
                   batch_size = 23,
                   return_losses=True,
                   return_weights=True,
                   seed=80718)
losses = train_info[0]
weights = train_info[1]
plt.plot(list(range(1000)),losses)
plt.show()



##### Predictions #####
def predict(X: ndarray,
            weights: Dict[str,ndarray]):
    '''Generate predictions from the step-by-step linear regression model
    '''
    N = np.dot(X,weights['W'])
    return N+weights['B']

preds = predict(X_test, weights) # weights = train_info[0]
plt.scatter(preds,y_test)
plt.plot(y_test,y_test)
plt.show()


##### Mean absolute error #####
def mae(preds: ndarray,actuals: ndarray)->float:
    '''Compute mean absolute error
    '''
    return np.mean(np.abs(preds-actuals))

##### Root mean squared error #####
def rmse(preds: ndarray, actuals: ndarray)->float:
    '''Compute RMSE
    '''
    return np.sqrt(np.mean(np.power(preds-actuals,2)))

print("Mean absolute error: ", mae(preds,y_test))
print("Root mean squared error: ", rmse(preds,y_test))


plt.scatter(X_test[:,6],y_test)
plt.show()