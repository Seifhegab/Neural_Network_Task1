import numpy as np
import pandas as pd
from sklearn import preprocessing
import random
from sklearn.model_selection import train_test_split


class SplitData:
    data = None

    def __init__(self, data, Class_list, Features_list):
        self.data = data
        self.Class_list = Class_list
        self.Features_list = Features_list

    def dataSplit(self):
        x1 = None
        x2 = None
        class1 = None
        class2 = None
        x = self.data.iloc[:, 0:5]
        y = self.data["Class"]

        if self.Class_list[0] == "BOMBAY":
            x1 = x.iloc[0:50, 0:5]
            class1 = self.data["Class"].iloc[0:50]
        elif self.Class_list[0] == "CALI":
            x1 = x.iloc[50:100, 0:5]
            class1 = self.data["Class"].iloc[50:100]
        elif self.Class_list[0] == "SIRA":
            x1 = x.iloc[100:160, 0:5]
            class1 = self.data["Class"].iloc[100:160]

        if self.Class_list[1] == "BOMBAY":
            x2 = x.iloc[0:50, 0:5]
            class2 = self.data["Class"].iloc[0:50]
        elif self.Class_list[1] == "CALI":
            x2 = x.iloc[50:100, 0:5]
            class2 = self.data["Class"].iloc[50:100]
        elif self.Class_list[1] == "SIRA":
            x2 = x.iloc[100:160, 0:5]
            class2 = self.data["Class"].iloc[100:160]

        # get the features
        x1_f1 = x1[self.Features_list[0]].to_numpy()
        x1_f2 = x1[self.Features_list[1]].to_numpy()
        x2_f1 = x2[self.Features_list[0]].to_numpy()
        x2_f2 = x2[self.Features_list[1]].to_numpy()

        col1 = pd.DataFrame({self.Features_list[0]: x1_f1, self.Features_list[1]: x1_f2})
        col2 = pd.DataFrame({self.Features_list[0]: x2_f1, self.Features_list[1]: x2_f2})
        class1 = pd.DataFrame({'Class': class1})
        class2 = pd.DataFrame({'Class': class2})

        X_train1, X_test1, y_train1, y_test1 = train_test_split(col1, class1, test_size=0.4, shuffle=True,
                                                                random_state=10)
        X_train2, X_test2, y_train2, y_test2 = train_test_split(col2, class2, test_size=0.4, shuffle=True,
                                                                random_state=10)

        X_train = pd.concat([X_train1, X_train2])
        X_test = pd.concat([X_test1, X_test2])
        y_train = pd.concat([y_train1, y_train2])
        y_test = pd.concat([y_test1, y_test2])

        return X_train,y_train,X_test,y_test