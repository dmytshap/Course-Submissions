"""
AI_Exercise5, part 1

Exercise 1. LSTM for generating text.
We want to use neural networks and natural language processing tools to generate text in a
specific context. The objective of this exercise is to generate songs lyrics specific to some
singer style.

"""
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
keras and tensorflow imports
'''
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import keras.utils


nltk.download('punkt')
nltk.download('punkt_tab')

MAX_WORD_SIZE = 3000
print("Start. Opening lyrics.")
lyrics = open("/eminem.txt","r")
lyrics = lyrics.read()

print("Lyrics read.")


# Splitting
lyrics = lyrics.split("\n")

print("Lyrics splitted.")

# Tokenization
print("Tokenizer initializing")
tokenizer = Tokenizer(num_words=MAX_WORD_SIZE, oov_token="<OOV>")
print("Tokenizer initialized")

print("Starting fitting on text")
tokenizer.fit_on_texts(lyrics)
print("Tokenizer fitted on text")

# index of words as dictionary, e.g: "word":1, "home":2, "window":3...
print("Word indexing initalizing")
word_index = tokenizer.word_index
print("Word indexing ran succesfully")

# Change text to integers
print("Texts to sequences initializing")
sequences = tokenizer.texts_to_sequences(lyrics)
print("Texts to sequences initialized sucesfully")
print(sequences)
print(word_index)

'''
Ngramming
'''
print("Ngram Start")
MAX_LINE_TOKENS = 30

ngrammed_sequences = []

for lyric in lyrics:
  token_list = tokenizer.texts_to_sequences(
      [lyric]
  )[0]

  if len(token_list) < 2:
    continue

  token_list = token_list[:MAX_LINE_TOKENS]

  for i in range(1, len(token_list)):
    n_gram_sequence = token_list[:i+1]
    ngrammed_sequences.append(n_gram_sequence)


print("Number of n-gram training sequences: ", len(ngrammed_sequences))
print("First 10 sequences")
print(ngrammed_sequences)

max_sequence_length = 0
for ngrammed_sequence in ngrammed_sequences:
  max_sequence_length = max(max_sequence_length, len(ngrammed_sequence))

print("Padding started")
padded_sequences = pad_sequences(
    ngrammed_sequences,
    maxlen=max_sequence_length,
    padding="pre")
print("Padded succesfully")
print(padded_sequences)

'''
Predictors and Labels, one-hot encoding
'''

predictors = padded_sequences[:, :-1]
labels = padded_sequences[:, -1]

vocab_size = MAX_WORD_SIZE

print(predictors.shape)
print(labels.shape)
print(labels)

print(padded_sequences[0])
print(labels[0])

'''
Data split for training and testing
'''

predictor_train, predictor_test, y_train, y_test = train_test_split(
    predictors,
    labels,
    test_size=0.2,
    random_state=42)

print(predictor_train.shape)
print(predictor_test.shape)
print(y_train.shape)
print(y_test.shape)

tf.keras.backend.clear_session()

'''
LSTM model
'''

model = tf.keras.Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=64,
        mask_zero=True
    ),
    LSTM(128),
    Dropout(0.2),
    Dense(
        vocab_size,
        activation="softmax"
    )
])

print()
model.build(
    input_shape=(None, max_sequence_length-1)
)

print("MODEL SUMMARY")
model.summary()

'''
MODEl COMPILE
'''

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["sparse_categorical_accuracy"]

)
print("Model compile finished.")

'''
Early stopping
'''

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
print("Early stopping finished.")

'''
Model training
'''

fitting = model.fit(
    predictor_train,
    y_train,
    validation_data=(predictor_test, y_test),
    epochs=30,
    batch_size=128,
    verbose=1,
    callbacks=[early_stopping]
)

'''
Index to word dictionary
'''

index_to_word_vocab = {}

for word, index in tokenizer.word_index.items():
  if index < vocab_size:
    index_to_word_vocab[index] = word

print(word_index)
print(word_index["i"])
print(index_to_word_vocab)
print(index_to_word_vocab[2])
print("Index -> word dictionary created")

'''
Word prediction
'''

def predict_next_word(seed_text):
  sequence = tokenizer.texts_to_sequences([seed_text])[0]
  sequence = pad_sequences([sequence],
                           maxlen=max_sequence_length-1,
                           padding="pre")
  probabilities = model.predict(sequence,
                                verbose=0)[0]
  probabilities[0] = 0.0

  predicted_id = int(np.argmax(probabilities))

  predicted_word = index_to_word_vocab.get(predicted_id)

  return predicted_word, probabilities[predicted_id]

print("Word prediction function end")

seed_text = "Stiffen up that upper  "

predicted_word, confidence = predict_next_word(seed_text)

print(f"Text to predict: {seed_text}")
print(f"Predicted next word: {predicted_word}")
print(f"Prediction probability {confidence}")

print(fitting.history)

'''
Show 5 next word predictions.
'''

sequence = tokenizer.texts_to_sequences([seed_text])[0]
sequence = pad_sequences([sequence],
                          maxlen=max_sequence_length-1,
                          padding="pre")
probabilities = model.predict(sequence,
                              verbose=0)[0]
probabilities[0] = 0.0

top_indices = np.argsort(probabilities)[-5:][::-1]

for index in top_indices:
  print(
      f"{index_to_word_vocab.get(index,'<OOV>'):20s}"
      f"{probabilities[index]:.4f}"
    )

def generate_longer_text(seed_text, number_of_words=30):
  generated_text = seed_text

  for i in range(number_of_words):
    predicted_word, confidence = predict_next_word(generated_text)

    if predicted_word == "<OOV>":
      break

    generated_text += (
        " " + predicted_word
    )
  return generated_text

seeds = [
    "This slickety ",
    "honestly futile ",
    "Enough rhymes ",
    "Morphed into ",
    "a sorceress",
    "Hey Slim, I drank a fifth",
    "a fifth",
    "And "
]

for seed in seeds:
  print(generate_longer_text(seed, number_of_words=30))
  print()


print("Stop for generating a lot of words")

plt.figure(figsize=(10,5))
plt.plot(
    fitting.history["loss"],
    label = "Training Loss"
)

plt.plot(
    fitting.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Sparse Categorical Cross Entropy")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(
    fitting.history["sparse_categorical_accuracy"],
    label = "Training Loss"
)

plt.plot(
    fitting.history["val_sparse_categorical_accuracy"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Next-Word Accuracy")
plt.title("LSTM Next-Word Prediction Accuracy")
plt.legend()
plt.grid(True)
plt.show()
