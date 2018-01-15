__author__ = 'Weiliang Guo'

"""
Implementing a simple word2vec model

References:
http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
https://github.com/golbin/TensorFlow-Tutorials/blob/master/04%20-%20Neural%20Network%20Basic/03%20-%20Word2Vec.py
"""

import tensorflow as tf
import numpy as np


# Pass in a txt file to the class constructor when initializing it.
class Word2Vec:

    def __init__(self, file_path='', skip_window=1,
                 epochs=10000, learning_rate=0.1, mini_batch_size=10, embedding_size=2,
                 num_sampled=5):

        # Initialize hyper-parameters
        self.f = file_path
        self.skip_window = skip_window
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.mini_batch_size= mini_batch_size
        self.embedding_size = embedding_size
        # num_sampled is used for NCE loss, and should be smaller than mini_batch_size
        self.num_sampled = num_sampled

        # Prepare data
        contents = open(self.f, 'r').readlines()
        self.word_sequence = " ".join(contents).split()
        word_list = " ".join(contents).split()
        # keep words without repetitions
        self.word_list = list(set(word_list))
        # {'colorless.': 0, 'set.': 1, 'nitwit': 2, 'Searching': 3, 'openly': 4, ...
        self.vocabulary = {w: i for i, w in enumerate(word_list)}
        self.voca_size = len(word_list)

        if not self.f.endswith('.txt'):
            print('Please pass in a txt file.')
            return

    def create_skip_grams(self):
        sw = self.skip_window
        ws = self.word_sequence
        voc = self.vocabulary
        skip_grams = []
        # i represents the index of a word in word sequence

        # target is the value of a key-value pair in the vocabulary
        for i in range(len(ws)):
            target = voc[ws[i]]
            if i - sw >= 0:
                lower = i - sw
            else:
                if i == 0:
                    lower = 1
                else:

                    lower = 0

            if i + sw + 1 <= len(ws):
                upper = i + sw + 1
            else:
                upper = len(ws)
            for c in range(lower, upper):
                if c != i:
                    context_word = voc[ws[c]]
                    skip_grams.append([target, context_word])
        return skip_grams

    # Fetch mini-batch randomly from skip-grams
    def fetch_random_batch(self):
        sg = self.create_skip_grams()
        random_inputs = []
        random_labels = []
        random_index = np.random.choice(range(len(sg)), self.mini_batch_size, replace=False)

        for i in random_index:
            random_inputs.append(sg[i][0])  # target
            random_labels.append([sg[i][1]])  # context word

        return random_inputs, random_labels

    def construct_neural_net(self):
        bs = self.mini_batch_size
        vs = self.voca_size
        es = self.embedding_size
        ns = self.num_sampled
        lr = self.learning_rate
        inputs = tf.placeholder(tf.int32, shape=[bs])
        labels = tf.placeholder(tf.int32, shape=[bs, 1])
        # Initialize embedding matrix, a.k.a. the input-to-hidden layer weight matrix
        embeddings = tf.Variable(tf.random_uniform([vs, es], -1.0, 1.0))

        selected_embeds = tf.nn.embedding_lookup(embeddings, inputs)

        nce_weights = tf.Variable(tf.random_uniform([vs, es], -1.0, 1.0))
        nce_biases = tf.Variable(tf.zeros([vs]))

        loss_op = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embeds, ns, vs))
        train_op = tf.train.AdamOptimizer(lr).minimize(loss_op)

        return inputs, labels, train_op, loss_op, embeddings

    def learn_embedding_matrix(self):
        with tf.Session() as sess:
            inputs, labels, train_op, loss_op, embeds = self.construct_neural_net()

            init = tf.global_variables_initializer()
            sess.run(init)

            for step in range(1, self.epochs + 1):
                batch_inputs, batch_labels = self.fetch_random_batch()

                _, loss_val = sess.run([train_op, loss_op],
                                       feed_dict={inputs: batch_inputs,
                                                  labels: batch_labels})

                # Assume loss is less than 3.0 is descent enough so  we may stop training at this point.
                if loss_val < 3.0:
                    print('A descent loss has been achieved at step '
                          + str(step), 'which is : ' + str(loss_val))
                    trained_embeddings = embeds.eval()
                    print(trained_embeddings)
                    return

                if step % 100 == 0:
                    print("Loss at step ", step, ": ", loss_val)

        # ----------------------------------------------
        # Loss at step  100 :  40.022263
        # Loss at step  200 :  13.611389
        # Loss at step  300 :  17.899164
        # Loss at step  400 :  39.618435
        # Loss at step  500 :  29.62997
        # A descent loss has been achieved at step 541 which is : 0.2846225
        # [[-0.16190553 -0.35054994]
        #  [ 0.18380451 -0.22422767]
        #  [-0.16630101 -0.77322507]
        #  ...
        #  [-0.01383805 -0.23162913]
        #  [ 2.8322225   2.7236242 ]
        #  [-0.85803866 -0.9154873 ]]


if __name__ == '__main__':
    w2v = Word2Vec(file_path='nlp_experiments/data/i_robot_asimov.txt')
    w2v.learn_embedding_matrix()
