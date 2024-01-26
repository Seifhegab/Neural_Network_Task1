import tkinter as tk
import numpy as np
import pandas as pd
from tkinter import *
from Preceptron_algorithm import Perceptron
from Adaline_Test import Adaline
from Preprocess import Preprocess
from Split_data import SplitData
import Plotting

Class_list = set()
Features_list = set()


# def reset_page():
#     # Clear the Entry widget
#     listbox_Class.selection_clear(0, tk.END)
#     listbox_Feature.selection_clear(0, tk.END)


def select_Class_list(event):
    selection1 = listbox_Class.curselection()
    for i in selection1:
        Class_list.add(listbox_Class.get(i))


def select_Features_list(event):
    selection2 = listbox_Feature.curselection()
    for i in selection2:
        Features_list.add(listbox_Feature.get(i))


def Change_Type():
    LearningRate = float(LearningRate_text.get("1.0", "end-1c"))
    Epochs = int(Epochs_text.get("1.0", "end-1c"))
    MSE = float(MSE_text.get("1.0", "end-1c"))
    bias = CheckVar1.get()
    Algorithm = RadioVar1
    x_train, y_train, x_test, y_test = Calling_Func()
    if Algorithm.get() == 1:
        perc = Perceptron(x_train, y_train,x_test,y_test, LearningRate, bias, Epochs, MSE)
        weight1, weight2,Bias = perc.Fitting()
        Plotting.plotting(x_train,np.array(list(Class_list)), Bias, weight1, weight2)
        perc.perceptronTest()
    elif Algorithm.get() == 2:
        Ada = Adaline(x_train, y_train,x_test,y_test, LearningRate, bias, Epochs, MSE)
        weight1, weight2, Bias = Ada.Fitting()
        Plotting.plotting(x_train,np.array(list(Class_list)), Bias, weight1, weight2)
        Ada.AdalineTest()


def Calling_Func():
    x = pd.read_csv("./Data/Dry_Bean_Dataset.csv").iloc[:, 0:5]
    y = pd.read_csv("./Data/Dry_Bean_Dataset.csv").iloc[:, 5:6]
    pre = Preprocess(x)
    pre.fill_null()
    preprocess_data = pre.Normalize()
    data = pd.concat([preprocess_data, y], axis=1)
    x_train, y_train, x_test, y_test = SplitData(data, np.array(list(Class_list)),  np.array(list(Features_list)))\
        .dataSplit()
    return x_train, y_train, x_test, y_test


m = tk.Tk()
m.title('Main Frame')
m.geometry("1000x1000")

Class_label = tk.Label(m, text='Please select two class :')
Class_label.pack()
Class_label.place(x=0, y=0)

listbox_Class = tk.Listbox(m, height=10, width=15, activestyle='dotbox', selectmode=tk.MULTIPLE)
listbox_Class.pack()
listbox_Class.place(x=200, y=0)
listbox_Class.insert(1, "BOMBAY")
listbox_Class.insert(2, "CALI")
listbox_Class.insert(3, "SIRA")
listbox_Class.bind("<<ListboxSelect>>", select_Class_list)

Feature_label = tk.Label(m, text='Please select two feature :')
Feature_label.pack()
Feature_label.place(x=0, y=200)

listbox_Feature = tk.Listbox(m, height=10, width=15, activestyle='dotbox', selectmode=tk.MULTIPLE)
listbox_Feature.pack()
listbox_Feature.place(x=200, y=200)
listbox_Feature.insert(1, "Area")
listbox_Feature.insert(2, "Perimeter")
listbox_Feature.insert(3, "MajorAxisLength")
listbox_Feature.insert(4, "MinorAxisLength")
listbox_Feature.insert(5, "roundnes")
listbox_Feature.bind("<<ListboxSelect>>", select_Features_list)

LearningRate_label = tk.Label(m, text='Please enter the learning rate')
LearningRate_label.pack()
LearningRate_label.place(x=0, y=400)
LearningRate_text = tk.Text(m, height=1, width=20)
LearningRate_text.pack()
LearningRate_text.place(x=200, y=400)

Epochs_label = tk.Label(m, text='Please enter the number of epochs')
Epochs_label.pack()
Epochs_label.place(x=0, y=450)
Epochs_text = tk.Text(m, height=1, width=20)
Epochs_text.pack()
Epochs_text.place(x=200, y=450)

MSE_label = tk.Label(m, text='Please enter the MSE threshold')
MSE_label.pack()
MSE_label.place(x=0, y=500)
MSE_text = tk.Text(m, height=1, width=20)
MSE_text.pack()
MSE_text.place(x=200, y=500)

Bias_label = tk.Label(m, text='Please choose if you want bias or not: ')
Bias_label.pack()
Bias_label.place(x=0, y=550)
CheckVar1 = IntVar()
Bias_Checkbox = tk.Checkbutton(m, text="Bias", activebackground="black", activeforeground="white", bd=0,
                               variable=CheckVar1,
                               onvalue=1, offvalue=0)
Bias_Checkbox.pack()
Bias_Checkbox.place(x=210, y=550)

Perceptron_label = tk.Label(m, text='Please choose your Algorithm: ')
Perceptron_label.pack()
Perceptron_label.place(x=0, y=600)
RadioVar1 = IntVar()
Perceptron_radiobutton = tk.Radiobutton(m, text="Perceptron algorithm", variable=RadioVar1, value=1)
Perceptron_radiobutton.pack()
Perceptron_radiobutton.place(x=210, y=600)
adaline_radiobutton = tk.Radiobutton(m, text="Adaline algorithm", variable=RadioVar1, value=2)
adaline_radiobutton.pack()
adaline_radiobutton.place(x=210, y=650)

generate_btn = tk.Button(m, text='Generate', width=15, command=Change_Type)
generate_btn.pack()
generate_btn.place(x=500, y=700)

# reset_button = tk.Button(m, text="Reset", command=reset_page)
# reset_button.pack()
# reset_button.place(x=600, y=700)

m.mainloop()
