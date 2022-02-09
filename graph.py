import numpy as np
import sys
from matplotlib import pyplot as plt

def read_results(filename):
    try:
        experiment_results = filename
    except Exception as e:
        print(e)
        exit(1)

    with open(experiment_results, "r") as f:
        lines = f.readlines()
    
    accuracies = {}

    index = -1
    keys  = []
    for line in lines:
        line = line.replace("\n","")
        try:
            num = float(line)
            if index > -1:
                accuracies[keys[index]].append(num)

        except:
            key = line.replace(" ", "_").replace(":","").lower()
            if key:
                keys.append(key)
                accuracies[key] = []
                index+=1

    return accuracies


def plot_results(results):
    # keys = list(results.keys())
    # cl = np.array(results[keys[1]])
    # fl = np.array(results[keys[2]])
    cl = np.array(results["centralized_learning_accuracies"])
    fl = np.array(results["federated_learning_accuracies"])
    x  = np.arange(fl.size)

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.plot(x, cl, "-b", label="Centralized")
    plt.plot(x, fl, "-r", label="FL")
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    results = read_results(sys.argv[1])
    # print(results.keys())
    plot_results(results)

