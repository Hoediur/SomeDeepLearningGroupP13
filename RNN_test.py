#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network for POS-tagging.
'''
from __future__ import division
import numpy as np
import theano
import theano.tensor as T
import lasagne
import re
import sys
from random import shuffle
import time
import gensim




# ############################## Data preparation ################################

unknown_token = "UNKNOWN_TOKEN"


def prepare_sents(corpus):
    """Preprocesses data: generate list of sentences, tags, vocabulary of tokens/tags,
    length of longest sentence in corpus"""

    sents = []
    sent = []
    tags = []
    sent_tags = []
    unique_tags = set()
    unique_tokens = set()
    longest_sent = 0

    with open(corpus) as f:
        for line in f:
            if line.strip(): # check if line not empty
                token = re.split(r'\t', line)[0]
                tag = re.split(r'\t', line)[1].strip()
                sent.append(token)
                sent_tags.append(tag)

                unique_tags.add(tag)
                unique_tokens.add(token)
            else:
                sents.append(sent)
                if len(sent) > longest_sent:
                    longest_sent = len(sent)
                tags.append(sent_tags)
                sent = []
                sent_tags = []

        if len(sent) > 0:
            sents.append(sent)
            if len(sent) > longest_sent:
                longest_sent = len(sent)
            tags.append(sent_tags)
            sent = []
            sent_tags = []

        unique_tokens = ["pad_value_rnn",unknown_token] + list(unique_tokens)
        unique_tags = ["pad_value_rnn"] + list(unique_tags)

    return sents, tags, unique_tokens, unique_tags, longest_sent

def pad_sent(sents,tags, longest_sent):
    for sent in sents:
        sent += ["pad_value_rnn"] * (longest_sent - len(sent))

    for tag in tags:
        tag += ["pad_value_rnn"] * (longest_sent - len(tag))

    return sents

def create_token_mappings(unique_tokens):
    token_to_index = dict([(w,i) for i,w in enumerate(unique_tokens)])
    return token_to_index

def create_tag_mappings(unique_tags):
    unique_tags = list(unique_tags)
    tag_to_index = dict([(t,i) for i,t in enumerate(unique_tags)])
    tag_to_vector = {}

    for i, tag in enumerate(unique_tags):
        #tag_to_vector[tag] = np.zeros(len(unique_tags))
        #tag_to_vector[tag][i] = 1
        tag_to_vector[tag] = i
    return tag_to_index, tag_to_vector

# words in corpus /training examples
N_DATA = 719530

# Number of epochs to train the net
NUM_EPOCHS = 10

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 1

NUM_UNITS = 1

NUM_OUTPUT_UNITS = 11


def build_network(X_train, unique_tags, unique_tokens, longest_sent, input_var=None):

    print("Building network ...")
    # shape = batch size, longest sent
    l_in = lasagne.layers.InputLayer(shape=(None, longest_sent), input_var=input_var)
    # l_em = lasagne.layers.EmbeddingLayer(l_in, input_size=len(unique_tokens), output_size=100, W=W)
    l_em = lasagne.layers.EmbeddingLayer(l_in, input_size=len(unique_tokens), output_size=100,  W = lasagne.init.Uniform(0.1))
    # l_ex = lasagne.layers.ExpressionLayer(l_in, lambda X: X, output_shape='auto')
    l_re = lasagne.layers.RecurrentLayer(l_em, N_HIDDEN, nonlinearity=lasagne.nonlinearities.sigmoid,
                                         mask_input=None)
    # l_out = lasagne.layers.DenseLayer(l_re, len(unique_tags), nonlinearity=lasagne.nonlinearities.softmax)
    l_out = lasagne.layers.DenseLayer(l_re, 130, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. This function returns only mini-batches
# of size `batchsize`. If the size of the data is not a multiple of `batchsize`,
# it will not return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize=1, shuffle=False):
    assert len(inputs) == len(targets)

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# ############################## Main program ################################

def main():

    # Load the dataset
    print "Loading data..."

    sents_train, tags_train, unique_tokens_train, unique_tags_train, longest_sent_train = prepare_sents(train_corpus)

    token_mappings = create_token_mappings(unique_tokens_train)
    tag_index_mappings, tag_vector_mappings = create_tag_mappings(unique_tags_train)

    pad_sent(sents_train, tags_train, longest_sent_train)
    print np.array(sents_train).shape
    print sents_train[0]
    X_train = np.asarray([[token_mappings[w] for w in sent] for sent in sents_train])

    y_train = np.asarray([[tag_vector_mappings[t] for t in sent_tags] for sent_tags in tags_train])
    #y_train = np.asarray([[6, 4, 4, 3, 0, 1, 8, 2, 6], [2, 9, 1, 2, 5, 2, 7, 6, 0]])
    # y_train = np.asarray(
    # [[[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    #   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #   [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    #   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]],
    #  [[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]])

    # print X_train
    # print y_train.shape
    sents_val, tags_val, unique_tokens_val, unique_tags_val, longest_sent_val = prepare_sents(val_corpus)

    pad_sent(sents_val, tags_val, longest_sent_val)
    X_val = np.asarray([[(token_mappings[w] if w in token_mappings else token_mappings[unknown_token])for w in sent ] for sent in sents_val ])
    y_val = np.asarray([[tag_vector_mappings[t] for t in sent_tags] for sent_tags in tags_val])
    #y_val = np.asarray([[6, 4, 4, 3, 0, 1, 8, 2, 6], [2, 9, 1, 2, 5, 2, 7, 6, 0]])


    # y_val = np.asarray(
    # [[[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    #   [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #   [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    #   [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #   [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]],
    #  [[0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    #         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #         [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.]]])

    sents_test, tags_test, unique_tokens_test, unique_tags_test, longest_sent_test = prepare_sents(test_corpus)
    pad_sent(sents_test,tags_test, longest_sent_test)
    X_test = np.asarray([[(token_mappings[w] if w in token_mappings else token_mappings[unknown_token]) for w in sent] for sent in sents_test ])
    y_test = np.asarray([[tag_vector_mappings[t] for t in sent_tags] for sent_tags in tags_test])

    #y_test = np.asarray([[6, 4, 4, 3, 0, 1, 8, 2, 6], [2, 9, 1, 2, 5, 2, 7, 6, 0]])

    input_var = T.lmatrix('inputs')
    # target_var = T.dtensor3('targets')
    target_var = T.dmatrix('targets')

    # # gensim 2D NumPy matrix
    # model = gensim.models.Word2Vec(sents_train, min_count=1)
    # # shape = (len(unique_tokens), 100)
    # W = model.syn0.astype("float64")
    # # print W.shape

    network = build_network(X_train, unique_tags_train, unique_tokens_train, longest_sent_train, input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize:
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=0), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(NUM_EPOCHS):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()


        print X_train.shape
        print y_train.shape

        for batch in iterate_minibatches(X_train, y_train, 1, shuffle=True):
            inputs, targets = batch
            print "TEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEST"
            train_err += train_fn(inputs, targets)
            train_batches += 1
            print train_batches

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 1, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, NUM_EPOCHS, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 1, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

if __name__ == '__main__':
    train_corpus = "de-train.txt"
    val_corpus = "de-dev.txt"
    test_corpus = "de-test.txt"
    print main()

