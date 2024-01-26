import numpy as np
import random
import pandas as pd
import Evaluation


class Adaline:
    w1 = None
    w2 = None
    X = None
    Y = None
    lr = 0.001
    epochs = 100
    mse_threshold = 0
    realClass = None
    feat1 = None
    feat2 = None
    bias = 0
    bias_Check = 0

    def __init__(self,X, Y,x_test,y_test, lr,bias_Check, epochs=100, mse_threshold=0):
        self.X = X
        self.Y = Y
        self.lr = lr
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.mse_threshold = mse_threshold
        self.bias_Check = bias_Check

    def labelClass(self):
        self.Y[:30] = 1
        self.Y[30:60] = -1

    def convertNumPy(self,X,Y):
        self.realClass = np.array(Y)
        self.feat1 = np.array(X.iloc[:, 0:1])
        self.feat2 = np.array(X.iloc[:, 1:2])

    def Training(self):
        for i in range(self.epochs):
            for x1, x2, yClass in zip(self.feat1, self.feat2, self.realClass):
                net1 = np.dot(self.w1, x1) + np.dot(self.w2, x2) + self.bias
                pred = self.Linear(net1)
                error = self.Compute_Error(yClass, pred)
                self.Update_Weights(error,x1,x2)
            mse = self.calc_mse()
            if mse <= self.mse_threshold:
                break

    def calc_mse(self):
        summation = 0
        for x1, x2, yClass in zip(self.feat1, self.feat2, self.realClass):
            net1 = np.dot(self.w1, x1) + np.dot(self.w2, x2) + self.bias
            pred = self.Linear(net1)
            error = self.Compute_Error(yClass, pred)
            error = pow(error,2)
            summation = summation + error
        mse = 0.5 * (summation / 2)
        return mse

    def Linear(self, val):
        return val

    def Compute_Error(self,actual,pred):
        error = actual - pred
        return error

    def Update_Weights(self, error,x1,x2):
        if error != 0:
            self.w1 += self.lr * error * x1
            self.w2 += self.lr * error * x2
            self.bias += self.lr * error

    def Generate_Weights_Bias(self):
        random.seed(10)
        # get the bias
        if self.bias_Check == 1:
            self.bias = round(random.random(), 6)
        else:
            self.bias = 0

        # get the weights
        self.w1 = round(random.random(), 6) * 0.01
        self.w2 = round(random.random(), 6) * 0.01

    def Fitting(self):
        self.Generate_Weights_Bias()
        self.labelClass()
        self.convertNumPy(self.X,self.Y)
        self.Training()
        return self.w1, self.w2, self.bias

    def AdalineTest(self):
        self.y_test[:20] = 1
        self.y_test[20:40] = -1
        y_pred = []
        self.convertNumPy(self.x_test,self.y_test)
        for x1, x2, yClass in zip(self.feat1, self.feat2, self.realClass):
            net1 = np.dot(self.w1, x1) + np.dot(self.w2, x2) + self.bias
            pred = self.Linear(net1)
            if pred >= 0:
                pred = 1
            elif pred < 1:
                pred = -1
            y_pred.append(pred)
        y_pred = pd.DataFrame({'Class': y_pred})

        # confusion matrix
        y_pred = y_pred.to_numpy()
        self.y_test = self.y_test.to_numpy()
        Evaluation.confusion_matrix(y_pred, self.y_test)

        # Accuracy
        Evaluation.calculate_accuracy(y_pred, self.y_test)


