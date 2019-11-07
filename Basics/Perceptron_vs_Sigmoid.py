# -*- coding: utf-8 -*-
"""

http://neuralnetworksanddeeplearning.com/chap1.html#exercise_263792

Small example to show difference between a perceptron and sigmoid neuron


For a Machine Learning model to learn we need a mechanism to adjust output of model by small adjustments in input. This example shows 
why a percptron is unsuitable for learning tasks and how sigmoid neuron is different.

"""
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns

B = 1
X = 1
N = 10
STEP = 1

W = [i for i in range(-N, N+1, STEP)]

def perceptron(w):
    
    z = w * X + B
    
    out = 0
    
    if z > 0:
        out =1
    else:
        out = 0
        
    return out
    
    
def sigmoid(w):
    
     
    z = w * X + B
    out = 1.0/(1.0+np.exp(-z))
    return out



y_perc = [perceptron(i) for i in W]
y_sig = [sigmoid(i) for i in W]


plt.plot(W, y_perc, '--o')
plt.plot(W, y_sig, '--X')

plt.title('Perceptron vs Sigmoid Neuron')
plt.xlabel('Input Weight')
plt.ylabel('Output Y')

plt.legend(['Perceptron', 'Sigmoid'], loc='upper left')

plt.show()


