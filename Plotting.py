import matplotlib.pyplot as plt
import numpy as np


def plotting(x,y,bias,weight1,weight2):
    np.random.seed(10)
    plt.scatter(x.iloc[0:30, 0:1], x.iloc[0:30, 1:2], c='blue', label=y[0])
    plt.scatter(x.iloc[30:60, 0:1], x.iloc[30:60, 1:2], c='red', label=y[1])
    feat1 = np.array(x.iloc[:, 0:1])
    feat2 = np.array(x.iloc[:, 1:2])
    xx = np.linspace(np.min(feat1), np.max(feat2), 100)
    y = -(bias + weight1 * xx) / weight2
    plt.plot(xx, y, color='green', label='Discrimination Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("training data")
    plt.legend()
    plt.show()
