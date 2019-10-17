# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:44:13 2019

@author: Amaan
"""

import numpy as np
import cupy as cp
import time
import timeit


### Numpy and CPU
s = time.time()
x_cpu = np.ones((100,1000,1000))
e = time.time()
print('Time to initialize in CPU/np', e - s)### CuPy and GPU
s = time.time()
x_gpu = cp.ones((100,1000,1000))
e = time.time()
print('Time to initialize in GPU/cp', e - s)


def initialize_test():
     
    ### Numpy and CPU
    s = timeit.default_timer()
    x_cpu = np.ones((300,1000,1000))
    e = timeit.default_timer()
    #print(e - s)### CuPy and GPU
    cpu_time = e - s
    print('Time to multiply in CPU/np', cpu_time)### CuPy and GPU
    
    s = timeit.default_timer()
    x_gpu = cp.ones((300,1000,1000))
    e = timeit.default_timer()
    gpu_time = e - s
    print('Time to Multiply in GPU/cp', gpu_time)
    
    print('GPU is {}X faster than CPU!!!!'.format( cpu_time/gpu_time))
    
    
initialize_test()       

def multiply_test(n=5):
    
    x_cpu = np.ones((300,1000,1000))
    x_gpu = cp.ones((300,1000,1000))

    ### Numpy and CPU
    s = timeit.default_timer()
    x_cpu *= n
    e = timeit.default_timer()
    #print(e - s)### CuPy and GPU
    cpu_time = e - s
    print('Time to multiply in CPU/np', cpu_time)### CuPy and GPU
    
    s = timeit.default_timer()
    x_gpu *= n
    e = timeit.default_timer()
    gpu_time = e - s
    print('Time to Multiply in GPU/cp', gpu_time)
    
    print('GPU is {}X faster than CPU!!!!'.format( cpu_time/gpu_time))
    
def complex_test(n=5):
    
    x_cpu = np.ones((300,1000,1000))
    x_gpu = cp.ones((300,1000,1000))

    ### Numpy and CPU
    s = timeit.default_timer()
    x_cpu *= 5
    x_cpu *= x_cpu
    x_cpu += x_cpu

    e = timeit.default_timer()
    #print(e - s)### CuPy and GPU
    cpu_time = e - s
    print('Time to multiply in CPU/np', cpu_time)### CuPy and GPU
    
    s = timeit.default_timer()
    x_gpu *= 5
    x_gpu *= x_gpu
    x_gpu += x_gpu
    e = timeit.default_timer()
    gpu_time = e - s
    print('Time to Multiply in GPU/cp', gpu_time)
    
    print('GPU is {}X faster than CPU!!!!'.format( cpu_time/gpu_time))
    
import numpy, cupy    
x = numpy.random.random((5000, 10000))
y = cupy.random.random((5000, 10000))

%timeit bool((numpy.sin(x) ** 2 + numpy.cos(x) ** 2 == 1).all())

%timeit bool((cupy.sin(y) ** 2 + cupy.cos(y) ** 2 == 1).all())


multiply_test()    
complex_test()
