import csv
import numpy as np
from scipy import optimize as op
import matplotlib.pyplot as plt

def sigmoid(Z):
    return (1 / (1 + np.exp(-Z))),Z

def relu(Z):
        return Z * (Z > 0),Z

def initialize_parameters(n_x, n_h, n_y):
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# def costfn(Y,A,lambd,theta):
#   a = Y*np.log(A) + (1-Y)*np.log(1-A)
#   cost = ((-1/m)*np.sum(a)) + (lambd/(2*m))*np.sum(theta**2)
#   return cost

# def gradientdesc(theta,A,Y,alph):
#   norows = theta.shape[0]
#   nocols = theta.shape[1]
#   grad = np.zeros((norows,nocols))
#   grad[0] = theta[0] - (alph/m)

def initialize_parameters_deep(layer_dims):
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def linear_forward(A, W, b):
    
    Z = np.dot(W,A)+b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
# A = np.array([[1.62434536, -0.61175641],[-0.52817175, -1.07296862],[0.86540763, -2.3015387]])
# W = np.array([[ 1.74481176, -0.7612069, 0.3190391]])
# b = np.array([[-0.24937038]])

# Z, linear_cache = linear_forward(A, W, b)
# print("Z = " + str(Z))

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache



# A_prev = np.array([[-0.41675785, -0.05626683],
#        [-2.1361961 ,  1.64027081],
#        [-1.79343559, -0.84174737]])
# W = np.array([[ 0.50288142, -1.24528809, -1.05795222]])
# b = np.array([[-0.90900761]])
# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))

# A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))

def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)
    A_prev = A 
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A_prev, W, b, "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches



def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = np.sum(np.sum((-1/m)*(np.dot(Y,np.log(AL).T) + np.dot((1.-Y),np.log(1.-AL).T))))
    cost = np.squeeze(cost)
    # assert(cost.shape == ())
    
    return cost

# Y = np.array([[1,1,1]])
# AL = np.array([[0.8,0.9,0.4]])
# print("cost = " + str(compute_cost(AL, Y)))

def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
   
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
   
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
    dA_prev_temp = grads["dA"+str(L-1)]
    
    for l in reversed(range(L-1)):
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache,'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        

    return grads

def predict(X, y, parameters):
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db"+str(l+1)]
    ### END CODE HERE ###
    return parameters

def give_row(n):
    a=[]
    if n==0:
        a=[1,0,0,0,0,0,0,0,0,0]
    elif n==1:
        a=[0,1,0,0,0,0,0,0,0,0]
    elif n==2:
        a=[0,0,1,0,0,0,0,0,0,0]
    elif n==3:
        a=[0,0,0,1,0,0,0,0,0,0]
    elif n==4:
        a=[0,0,0,0,1,0,0,0,0,0]
    elif n==5:
        a=[0,0,0,0,0,1,0,0,0,0]
    elif n==6:
        a=[0,0,0,0,0,0,1,0,0,0]
    elif n==7:
        a=[0,0,0,0,0,0,0,1,0,0]
    elif n==8:
        a=[0,0,0,0,0,0,0,0,1,0]
    elif n==9:
        a=[0,0,0,0,0,0,0,0,0,1]
    return a


#now we call all the functions


with open('train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        flag=0
        a=[]
        for row in csv_reader:
            if flag==0:
                flag=1
                continue
            a.append(row)
            line_count+=1
            if line_count==1000:
                break
temp = np.array(a).astype(np.float32)
train_x = temp[:,1:].T
y_temp = temp[:,:1]
train_y = (y_temp).T
yt = []
for i in range(1000):
    if train_y[0][i] != 2:
        train_y[0][i] = 0
    else:
        train_y[0][i] = 1
    # yt.append(give_row(train_y[0][i]))
# train_y=np.asarray(yt).astype(np.float32).T
with open('train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        flag=0
        a=[]
        for row in csv_reader:
            if flag==0:
                flag=1
                continue
            line_count+=1
            if line_count<1000:
                continue
            if line_count==1500:
                break
            a.append(row)
temp = np.array(a).astype(np.float32)
test_x = temp[:,1:].T
y_temp = temp[:,:1]
test_y = (y_temp).T
yt = []
for i in range(500):
    # yt.append(give_row(test_y[0][i]))
    if test_y[0][i] != 2:
        test_y[0][i] = 0
    else:
        test_y[0][i] = 1
# test_y = np.asarray(yt).astype(np.float32).T
# print ("train_x's shape: " + str(train_x.shape))
# print ("test_x's shape: " + str(test_x.shape))


n_x = 784
n_h = 20
n_y = 1
layers_dims = (n_x, n_h, n_y)

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 1500, print_cost=False):
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        # print(A2.shape[0],A2.shape[1])
        cost = compute_cost(A2, Y)
       
        t1 = np.divide(Y,A2)
        t2 = np.divide(1 - Y, 1 - A2)
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
       
        parameters = update_parameters(parameters, grads, learning_rate)
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

# print(train_y)

parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2200, print_cost=True)

predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)