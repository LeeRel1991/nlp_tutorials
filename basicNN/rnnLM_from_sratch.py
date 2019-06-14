"""
@author: rui.li
@contact: xx@xx.com
@software: PyCharm
@file: rnn.py
@time: 2019/6/4 15:45
"""
import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
# from utils import *
# from rnn_theano import RNNTheano

nltk.download("book")

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '8000'))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


def get_dataset(csv_file):
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(csv_file, 'rb') as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return X_train, y_train


class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        """

        Args:
            word_dim: input单词维度，此处采用one-hot作为输入，因此等于词汇大小|V|
            hidden_dim: 隐层维度，即隐层的word vector表示。补充：隐层神经元数=输入word数
            bptt_truncate:
        """
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        # U: input x 到 隐层S的权值矩阵，x: [T, |V|], T 表示单词个数，也即时间状态数，S：[T, hidden_dim],
        # V: 隐层S到输出O的权值矩阵
        # W: 隐层 S(t-1) 到S(t)的权值矩阵
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        """
        rnn 前向传播
        Args:
            x: [T, |V|]， T表示单词个数，每一行表示一个单词的one-hot编码

        Returns:
            每个隐层神经元的输出（一共T个神经元，与单词一一对应，[T, |V|]），以及每个隐层神经元的状态([T, hiddem_dim])
        """
        # The total number of time steps
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))
        # Note: 不能直接用矩阵相乘，因为S中的每一行（每一个时间戳）依赖于上一行的结果，所以只能一行一行的遍历计算
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = self.softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def softmax(self, x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence...
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            # We only care about our prediction of the "correct" words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L

    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        # For each output backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step - 1])
                dLdU[:, x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dLdU, dLdV, dLdW]


def gradient_check(model, x, y, h=0.001, error_threshold=0.01):
    # Calculate the gradients using backpropagation. We want to checker if these are correct.
    bptt_gradients = model.bptt(x, y)
    # List of all parameters we want to check.
    model_parameters = ['U', 'V', 'W']
    # Gradient check for each parameter
    for pidx, pname in enumerate(model_parameters):
        # Get the actual parameter value from the mode, e.g. model.W
        parameter = operator.attrgetter(pname)(self)
        print("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
        # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            # Save the original value so we can reset it later
            original_value = parameter[ix]
            # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
            parameter[ix] = original_value + h
            gradplus = model.calculate_total_loss([x], [y])
            parameter[ix] = original_value - h
            gradminus = model.calculate_total_loss([x], [y])
            estimated_gradient = (gradplus - gradminus) / (2 * h)
            # Reset parameter to original value
            parameter[ix] = original_value
            # The gradient for this parameter calculated using backpropagation
            backprop_gradient = bptt_gradients[pidx][ix]
            # calculate The relative error: (|x - y|/(|x| + |y|))
            relative_error = np.abs(backprop_gradient - estimated_gradient) / (
                        np.abs(backprop_gradient) + np.abs(estimated_gradient))
            # If the error is to large fail the gradient check
            if relative_error > error_threshold:
                print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                print("+h Loss: %f" % gradplus)
                print("-h Loss: %f" % gradminus)
                print("Estimated_gradient: %f" % estimated_gradient)
                print("Backpropagation gradient: %f" % backprop_gradient)
                print("Relative Error: %f" % relative_error)
                return
            it.iternext()
        print("Gradient check for parameter %s passed." % (pname))


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("./data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time),
                                         model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1



def main():
    # X_train, y_train = get_dataset('../data/lm/reddit-comments-2015-08.csv')
    # print(X_train.shape, y_train.shape)

    np.random.seed(10)
    model = RNNNumpy(vocabulary_size)
    x = np.array([[0, 51, 27, 16, 10, 856, 53, 25, 34, 69]])
    o, s = model.forward_propagation(X_train[10])

    # model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
    # t1 = time.time()
    # model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
    # t2 = time.time()
    # print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))
    #
    # if _MODEL_FILE != None:
    #     load_model_parameters_theano(_MODEL_FILE, model)
    #
    # train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

if __name__ == "__main__":
    main()