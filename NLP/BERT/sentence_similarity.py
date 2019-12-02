# -*- coding: utf-8 -*-
"""
https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15
Created on Sun Dec  1 14:26:45 2019

@author: Amaan
"""


import tensorflow_hub as hub
import numpy as np
import seaborn as sns;

#embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")
embed = hub.load("model/tf2")


# Compute a representation for each message, showing various lengths supported.
word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
messages = [word, sentence, paragraph]

message_embeddings = embed(messages)["outputs"]
message_embeddings

for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        print("Message: {}".format(messages[i]))
        print("Embedding size: {}".format(len(message_embedding)))
        message_embedding_snippet = ", ".join((str(x) for x in        message_embedding[:3]))
        print("Embedding[{},...]\n".
                   format(message_embedding_snippet))
        


# Sentence Similarity
messages = [
    "we are sorry for the inconvenience",
    "we are sorry for the delay",
    "we regret for your inconvenience",
    "we don't deliver to baner region in pune",
    "we will get you the best possible rate"
]

encoding_matrix = embed(messages)["outputs"]
corr = np.inner(encoding_matrix, encoding_matrix)
ax = sns.heatmap(corr, annot=True)


# Sentence Similarity
messages = [
    "RBI increased repo rate by 0.25",
    "RBI increased repo rate by 25 basis points",
    "Repo rate was increased by 0.25",
    "Apple announced quarterly results",
    "Reserve bank of Scotland kept rates unchanged",
    "Ganga is biggest river of India"

]

encoding_matrix = embed(messages)["outputs"]
corr = np.inner(encoding_matrix, encoding_matrix)
ax = sns.heatmap(corr, annot=True)