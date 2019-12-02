# -*- coding: utf-8 -*-
"""
https://github.com/UKPLab/sentence-transformers/blob/master/examples/application_semantic_search.py
Created on Sun Dec  1 14:26:45 2019

@author: Amaan
"""


import tensorflow_hub as hub
import numpy as np
import seaborn as sns;
import scipy.spatial
#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
embed = hub.load("model/tf2")

# Corpus with example sentences
corpus = ['A man is eating a food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.'
          ]

corpus_embeddings = embed(corpus)["outputs"]

# Query sentences:
queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']
query_embeddings = embed(queries)["outputs"]

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 3
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop {} most similar sentences in corpus:".format(closest_n))

    for idx, distance in results[0:closest_n]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))

import numpy as np
import scipy

a = np.random.normal(size=(10,30))
b = np.random.normal(size=(3,30))

dist2 = scipy.spatial.distance.cdist(b,a,"cosine") # pick the appropriate distance metric 
d3 = dist2[0]
