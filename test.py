# Importing packages and loading embeddings.

import os
import numpy as np
import pandas as pd
import string
import json
    
embeddings_index = {}
f = open(r'embeddings/glove.840B.300d/glove.840B.300d.txt', encoding='utf8')
for line in f:
    values = line.split()
    word = ''.join(values[:-300])
    coefs = np.array([np.asarray(values[-300:], dtype='float32')])
    embeddings_index[word] = coefs
f.close()

print(len(embeddings_index))

# Loading data set into memory.

df = pd.read_csv('train.csv')
df.info()

# Separating by the labels

df_1 = df.loc[df['target'] == 1]
df_0 = df.loc[df['target'] == 0]

# Picking random samples from the abundant class.

df_0_randomsamples = df_0.sample(n=121215)
# df_0_randomsamples.head()

#Merging into one. (Ratio 1:1.5)

frames = [df_1, df_0_randomsamples]
result = pd.concat(frames)
df_balanced = result.sample(frac=1).reset_index(drop=True)
df_balanced.head()

# Converting df to numpy arrays.

data = df_balanced['question_text'].values
labels = df_balanced['target'].values


print(len(data))
print(len(labels))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

MAX_NB_WORDS = 2195892
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data)
sequences = tokenizer.texts_to_sequences(data)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_SEQUENCE_LENGTH = 200
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set

VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
nb_test_samples = int(TEST_SPLIT * data.shape[0])

x_val = data[:nb_validation_samples]
y_val = labels[:nb_validation_samples]
x_test = data[nb_validation_samples:nb_validation_samples+nb_test_samples]
y_test = labels[nb_validation_samples:nb_validation_samples+nb_test_samples]
x_train = data[nb_validation_samples+nb_test_samples:]
y_train = labels[nb_validation_samples+nb_test_samples:]

EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Embedding
from keras.models import *
from keras.layers import *
from keras.optimizers import SGD

def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv

model_conv = create_conv_model()

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


model = KerasClassifier(build_fn=create_conv_model, verbose=0)
# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(data, labels)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))