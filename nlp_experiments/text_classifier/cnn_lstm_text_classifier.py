__author__ = 'Weiliang Guo'
"""
References:
http://colah.github.io/posts/2015-08-Understanding-LSTMs/
https://machinelearningmastery.com/sequence-classification-text_classifier-recurrent-neural-networks-python-keras/
"""

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

# The weights of the model will be initialized with random numbers.
# By fixing a random seed, whenever we run run the code, if the given data and parameters are the same, then the output
# will always be the same.
numpy.random.seed(7)


class CnnLstmTextClassifier:
    # Pass in hyper-parameters
    def __init__(self, top_words=5000, max_txt_len=500, embed_vector_len=32,
                 cnn_filters=32, cnn_kernel_size=3, cnn_padding='same', cnn_activation='relu',
                 cnn_pool_size=2, num_memory_cells=100, dropout=0.2, recurrent_dropout=0.2,
                 activation='sigmoid', loss='binary_crossentropy', optimizer='adam',
                 metrics='accuracy', epochs=3):

        # The words have been replaced by integers that indicate the ordered frequency of each word in the dataset.
        # The sentences which are with varied lengths in each review are therefore comprised of a sequence of integers.
        #
        # Make a vocabulary out of the training set, which contains all distinct words,
        # these words are ordered by their frequencies (occurences) in the training set.
        #
        # X_train, y_train, X_test, y_test are all numpy arrays.
        #
        # shapes of X_train and y_train are (NUM_SENTENCES_IN_TRAINING_SET,),
        # one element(a sentence with varied length) in X_train is represented as [1, 12, 62, 306, ..., 121, 4, 2, 130, 56],
        # Here the numbers are words replaced with their corresponding  ordered frequencies.
        #
        # one element(a class label) of y_train is a numpy int64 integer , 1 or 0 if it's a binary classification.
        #
        # shapes of X_test and y_test are (NUM_SENTENCES_IN_TEST_SET,), elements of them are
        # represented the same as of X_train, y_train
        #
        # TODO: this function shall be rewritten to accept arbitrary text data rather than keras's built-in imdb data
        # load the dataset but only keep the top n words, zero the rest
        self.top_words = top_words
        (X_train, self.y_train), (X_test, self.y_test) = imdb.load_data(num_words=self.top_words)
        self.X_train = sequence.pad_sequences(X_train, maxlen=max_txt_len)
        self.X_test = sequence.pad_sequences(X_test, maxlen=max_txt_len)

        # The words have been replaced by integers that indicate the ordered frequency of each word in the dataset.
        # Training texts with varied lengths are therefore comprised of a sequence of integers.
        #
        # Make a vocabulary out of the training set, which contains all distinct words,
        # these words are ordered by their frequencies (occurrences) in the training set.
        #
        # ==== hyper-parameters for LSTM input layer of embedding ==== #
        # truncate and pad input sequences to be always equal to max_txt_len
        self.max_txt_len = max_txt_len
        # Embedding converts the sequence to a feature vector with fixed length.
        self.embed_vector_len = embed_vector_len

        # ==== hyper-parameters for LSTM input layer of CNN ==== #
        self.cnn_filters = cnn_filters
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_padding = cnn_padding
        self.cnn_activation = cnn_activation
        self.cnn_pool_size = cnn_pool_size

        # ==== hyper-parameters for LSTM memory cells ==== #
        # LSTM is theoretically a single-layer neural net, but with
        # "recurrent" feature, it behaves as if with multiple layers
        # which each layer is of a memory cell(just like one neuron) with same structure.
        self.num_memory_cells = num_memory_cells
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.activation = activation

        # ==== hyper-parameters for training LSTM model ==== #
        self.loss = loss
        # There are many available optimization techniques such as SGD, Adam or RMSProp,
        # we choose Adam because it usually achieves better performance than other ones.
        self.optimizer = optimizer
        self.metrics = [metrics]
        self.epochs = epochs
        self.batch_size = 64

    def create_lstm_model(self):
        model = Sequential()

        # ==== Adding LSTM input layer of embedding ==== #
        model.add(Embedding(self.top_words, self.embed_vector_len, input_length=self.max_txt_len))

        # ==== Adding LSTM input layer of CNN ==== #
        # This layer Conv1D creates a convolution kernel that is convolved with the
        # layer input over a single spatial (or temporal) dimension to produce a tensor of outputs.
        #
        # CNN is exellent at capturing spatial information so here introduce it to further enhance the feature vector
        #
        # One filter captures one feature. Here kernel is also called filter.
        #
        # padding: "same"` results in padding the input such that
        # the output has the same length as the original input
        #
        # The constant gradient of ReLUs results in faster learning.
        # It also results in sparse representations which seem to be more beneficial than dense representations.
        model.add(Conv1D(filters=self.cnn_filters, kernel_size=self.cnn_kernel_size,
                         padding=self.cnn_padding, activation=self.cnn_activation))
        model.add(MaxPooling1D(pool_size=self.cnn_pool_size))

        # ==== Adding LSTM hidden layer ==== #
        # dropout is a method to comabt over-fitting.
        #
        # dropout: Float between 0 and 1.  Fraction of the units
        # to drop for the linear transformation of the inputs.
        #
        # recurrent_dropout: Float between 0 and 1. Fraction of the units
        # to drop for the linear transformation of the recurrent state.
        model.add(LSTM(self.num_memory_cells, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout))

        # ==== Adding LSTM output layer ==== #
        # It's just your regular densely-connected NN layer.
        # Dimensionality of the output space is 1.
        # Use sigmoid activation function if it's a binary classificaton otherwise use softmax.
        model.add(Dense(1, activation=self.activation))
        return model

    def train_lstm_model(self):
        model = self.create_lstm_model()
        # Compiling the model to be used by the backend  TensorFlow, CNTK, or Theano
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        print(model.summary())
        model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=self.batch_size)
        return model

    def evaluate_lstm_model(self):
        model = self.train_lstm_model()
        # Final evaluation of the model
        scores = model.evaluate(self.X_test, self.y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    c = CnnLstmTextClassifier()
    c.evaluate_lstm_model()