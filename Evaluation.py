import numpy as np


def confusion_matrix(pred, real):
    conf_matrix = np.zeros((2,2))
    for i in range(len(pred)):
        if int(pred[i]) == 1 and int(real[i]) == 1:
            conf_matrix[0,0] += 1
        elif int(pred[i]) == 1 and int(real[i]) == -1:
            conf_matrix[0,1] += 1
        elif int(pred[i]) == -1 and int(real[i]) == 1:
            conf_matrix[1,0] += 1
        elif int(pred[i]) == -1 and int(real[i]) == -1:
            conf_matrix[1,1] += 1
    print("The confusion matrix = \n",conf_matrix)


def calculate_accuracy(pred, real):
    true_predictions = 0
    Sum_predictions = len(pred)
    for index_true, index_pred in zip(pred, real):
        if index_true == index_pred:
            true_predictions += 1
    accuracy = true_predictions / Sum_predictions
    print("Accuracy Equal = ", accuracy * 100,"%")

