import numpy as np
import pandas as pd

import nltk
import keras
import tensorflow as tf

from nltk import word_tokenize
from nltk import ngrams

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.utils


nltk.download('punkt')
nltk.download('punkt_tab')


'''
Start
'''

print("start")

lyrics = open("Exercise5/Song_Lyrics/eminem.txt","r")
lyrics = lyrics.read()

print("lyrics read")

# Splitting
lyrics = lyrics.split("\n")


# Tokenization
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(lyrics)

# index of words as dictionary, e.g: "word":1, "home":2, "window":3...
word_index = tokenizer.word_index

# Change text to integers
sequences = tokenizer.texts_to_sequences(lyrics)

## ngram
print("Ngram Start")
ngrammed_sequences = []

# Add ngrams to list.
for sequence in sequences:
    print("TEST", sequence)
    for i in range(2, len(sequence) + 1): # We do not need first word, so we start from index 2.
        ngrammed_sequences.append(sequence[:i]) # Adds ngrams to list -> e.g sequence [5,11,8,43] become [5,11], [5,11,8], [5,11,8,43]
        
# Find longest ngram
longest_ngram = 0
for ngrammed_sequence in ngrammed_sequences:
    longest_ngram = max(longest_ngram, len(ngrammed_sequence))

print(longest_ngram)

# ngramed_sequences = tf.strings.ngrams(tokenizer, 2).numpy()



# Padding
padded = pad_sequences(sequences)
