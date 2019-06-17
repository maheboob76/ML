# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 05:29:05 2019

@author: Amaan
"""

import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib tk

CHUNK = 1024 * 4 #samples per frame
FORMAT = pyaudio.paInt16 #audio format (bytes per sample?)
CHANNELS = 1 # single channel for microphone
RATE = 44100 # samples per second

p = pyaudio.PyAudio()

stream = p.open(
        format = FORMAT,
        channels = CHANNELS,
        rate = RATE,
        input = True,
        output = True,
        frames_per_buffer = CHUNK
        )


fig, ax = plt.subplots()

x = np.arange(0, 2 * CHUNK, 2)
line,  = ax.plot(x, np.random.rand(CHUNK))
ax.set_xlim(0, CHUNK)
ax.set_ylim(0, 255)

#ax.plot(data_int, '-')
#plt.show()


while True:  
    data = stream.read(CHUNK)
    data_int = np.array( struct.unpack( str(2 * CHUNK) + 'B', data), dtype='b')[::2] + 128
    line.set_ydata(data_int)
    fig.canvas.draw()
    fig.canvas.flush_events()