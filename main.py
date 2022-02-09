import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import threading
import datetime

#way to make unique training data for each thread
def worker_train(W1, b1, W2, b2, X, Y, alpha, iterations, lock): #training on threads
    for _ in range(iterations):
        Z1, A1, Z2, A2     = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2     = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

    predictions = get_predictions(A2)
    #write results to central location (for aggregation)
    with lock:
        global weights_and_biases
        global accuracies
        accuracy = get_accuracy(predictions, Y)
        print(f"\t\t\t{accuracy} after {iterations} iterations", flush=True)
        weights_and_biases.append((W1, b1, W2, b2)) #append weights and biases tuple
        accuracies.append(accuracy)

def aggregate_params(weights_n_biases, accs):
    W1_list,b1_list,W2_list,b2_list = list(),list(),list(),list()

    for wbs,acc in zip(weights_n_biases,accs):
        W1, b1, W2, b2 = wbs
        W1_list.append(W1)
        b1_list.append(b1)
        W2_list.append(W2)
        b2_list.append(b2)
    
    w1_sum,b1_sum,w2_sum,b2_sum, = 0,0,0,0
    for w1,b1,w2,b2,acc in zip(W1_list,b1_list,W2_list,b2_list,accs):
        w1_sum += w1 * acc
        b1_sum += b1 * acc
        w2_sum += w2 * acc
        b2_sum += b2 * acc

    W1 = w1_sum/len(W1_list)
    b1 = b1_sum/len(b1_list)
    W2 = w2_sum/len(W2_list)
    b2 = b2_sum/len(b2_list)
    return W1, b1, W2, b2

def start_workers(workers):
    for worker in workers:
        worker.start()
    
    for worker in workers:
        worker.join()

def split_data(data):    
    le_split   = int(data.shape[0]/np.random.randint(2,high=16))
    data_test  = data[0:le_split].T
    data_train = data[le_split:]
    return data_test, data_train
        
def create_workers(data_train, num_workers, params):
    W1, b1, W2, b2 = params
    m, n           = data_train.shape
    
    le_split = int(m/np.random.randint(2,high=16))
    
    workers    = list()
    lock       = threading.Lock()
    data_reset = np.copy(data_train)
    
    for _ in range(num_workers):
        alpha        = np.random.random_sample()#np.random.randint(0.1,high=0.5)
        k_iterations = np.random.randint(1000, high=1500)
        data_train   = data_reset[le_split:]
        np.random.shuffle(data_train)

        rand_num   = np.random.randint(2,high=16)
        data_train = data_train[:int(m/rand_num), :].T
        Y_train    = data_train[0]  #labels
        X_train    = data_train[1:] #image pixels
        X_train    = X_train / 255.
        
        cW1, cb1, cW2, cb2 = np.copy(W1), np.copy(b1), np.copy(W2), np.copy(b2)
        
        thread = threading.Thread(target=worker_train, args=(cW1, cb1, cW2, cb2, X_train, Y_train, alpha, k_iterations, lock))
        workers.append(thread)
    
    return workers
    
def read_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    return data

######################################################

def format_data(filename):
    data = pd.read_csv(filename)
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data) # shuffle before splitting into dev and training sets

    le_split  = int(m/np.random.randint(2,high=16))
    data_test = data[0:le_split].T
    Y_test    = data_test[0]
    X_test    = data_test[1:n]
    X_test    = X_test / 255.

    data_train = data[le_split:m].T
    Y_train    = data_train[0]
    X_train    = data_train[1:n]
    X_train    = X_train / 255.
    
    return (Y_test, X_test, Y_train, X_train)

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y                       = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y                       = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m         = Y.size
    one_hot_Y = one_hot(Y)
    dZ2       = A2 - one_hot_Y
    dW2       = 1 / m * dZ2.dot(A1.T)
    db2       = 1 / m * np.sum(dZ2)
    dZ1       = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1       = 1 / m * dZ1.dot(X.T)
    db1       = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, params, alpha, iterations):
    W1, b1, W2, b2 = params
    for _ in range(iterations):
        Z1, A1, Z2, A2     = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2     = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
    predictions            = get_predictions(A2)
    #         print(f"Iteration: {i}\n{get_accuracy(predictions, Y)}")
    #         print(get_accuracy(predictions, Y))
    accuracy = get_accuracy(predictions, Y)
    print(f"\t\t{accuracy} after {iterations} iterations", flush=True)

    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction    = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label         = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

#FL
weights_and_biases,accuracies = list(),list()
def run_federated_learning(d_test, d_train, W1, b1, W2, b2):
    for _ in range(10):
        global weights_and_biases 
        global accuracies
        weights_and_biases,accuracies = list(),list()
        # workers = create_workers(d_train, 5, (W1, b1, W2, b2))
        workers = create_workers(d_train, np.random.randint(5,high=15), (W1, b1, W2, b2))
        print(f"\t\tCreated {len(workers)} workers", flush=True)
        start_workers(workers)
        W1, b1, W2, b2 = aggregate_params(weights_and_biases,accuracies)
        # print(f"Updated params [iteration {j}]:\n",W1, b1, W2, b2)
        # print(f"Updated params [iteration {j}]:\n",b1)
        # print(f"Iteration {j} complete...")

    Y_test = d_test[0]
    X_test = d_test[1:d_test.shape[1]]
    X_test = X_test / 255.


    # print("Predictions")
    dev_predictions = make_predictions(X_test, W1, b1, W2, b2)
    test_accuracy   = get_accuracy(dev_predictions, Y_test)
    # print(test_accuracy)
    return test_accuracy

#Centralized
def run_centralized_learning(d_test, d_train, W1, b1, W2, b2):
    data_reset = np.copy(d_train)
    m, n       = d_train.shape

    for _ in range(10):
        alpha        = np.random.random_sample()
        k_iterations = np.random.randint(1000, high=1500)
        le_split     = int(m/np.random.randint(2,high=16))
        data_train   = data_reset[le_split:]
        np.random.shuffle(data_train)
        rand_num   = np.random.randint(2,high=16)
        data_train = data_train[:int(m/rand_num), :].T
        Y_train    = data_train[0]  #labels
        X_train    = data_train[1:] #image pixels
        X_train    = X_train / 255.

        W1, b1, W2, b2 = gradient_descent(X_train, Y_train, (W1, b1, W2, b2), alpha, k_iterations)
        # print(f"Iteration {i} complete...")


    Y_test = d_test[0]
    X_test = d_test[1:d_test.shape[1]]
    X_test = X_test / 255.

    # print("Predictions")
    dev_predictions = make_predictions(X_test, W1, b1, W2, b2)
    test_accuracy   = get_accuracy(dev_predictions, Y_test)
    # print(test_accuracy)
    return test_accuracy

if __name__ == "__main__":
    run_time                             = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    centralized_accuracies,fl_accuracies = list(),list()
    for epoch in range(2):
        print(f"TESTING ITERATION {epoch} START", flush=True)
        d               = read_data("./train.csv")
        W1, b1, W2, b2  = init_params()
        d_test, d_train = split_data(d)

        print("\tCentralized training start", flush=True)
        cl_accuracy = run_centralized_learning(d_test, d_train, np.copy(W1), np.copy(b1), np.copy(W2), np.copy(b2))
        print("\tCentralized training end", flush=True)

        print("\tFL training start", flush=True)
        fl_accuracy = run_federated_learning(d_test, d_train, np.copy(W1), np.copy(b1), np.copy(W2), np.copy(b2))
        print("\tFL training end", flush=True)
        
        centralized_accuracies.append(cl_accuracy)
        fl_accuracies.append(fl_accuracy)
        print(f"TESTING ITERATION {epoch} END\n", flush=True)
        
    cl = np.array(centralized_accuracies)
    fl = np.array(fl_accuracies)

    cl = np.insert(cl,0,0)
    fl = np.insert(fl,0,0)
    x  = np.arange(fl.size)

    with open(f"cl_vs_fl_{run_time}.txt", "w") as f:
        f.write("Centralized Learning accuracies:\n")
        for value in cl:
            f.write(f"{value}\n")
        
        f.write("\nFederated Learning accuracies:\n")
        for value in fl:
            f.write(f"{value}\n")
    

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.plot(x, cl, "-b", label="Centralized")
    plt.plot(x, fl, "-r", label="FL")
    plt.legend(loc="lower right")
