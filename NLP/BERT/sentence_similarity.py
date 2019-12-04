# -*- coding: utf-8 -*-
"""
https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15
Created on Sun Dec  1 14:26:45 2019

@author: Amaan
"""


import tensorflow_hub as hub
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
embed = hub.load("model/tf2/universal-sentence-encoder-v3")




# Sentence Similarity
messages = [
    "EPS estimates for APPL is expected to increase by 20 bps",
    "ROI estimates for APPL is expected to increase by 9%",
    "ROI estimates for GOOG is expected to increase by 9%",
    "ROI estimates for GOOG is expected to increase by 2%"

]

encoding_matrix = embed(messages)["outputs"]
corr = np.inner(encoding_matrix, encoding_matrix)
ax = sns.heatmap(corr, annot=True)
plt.show()