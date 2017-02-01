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



train_corpus = sys.argv[1]
val_corpus = sys.argv[2]
# test_corpus = sys.argv[3]

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
    print "Reading corpus file..."
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
        unique_tokens = [unknown_token] + list(unique_tokens)
    return sents, tags, unique_tokens, unique_tags, longest_sent

def pad_sent(sents, longest_sent):
    for sent in sents:
        sent += [unknown_token] * (longest_sent - len(sent))
    return sents

def create_token_mappings(unique_tokens):
    token_to_index = dict([(w,i) for i,w in enumerate(unique_tokens)])
    return token_to_index

def create_tag_mappings(unique_tags):
    unique_tags = list(unique_tags)
    tag_to_index = dict([(t,i) for i,t in enumerate(unique_tags)])
    return tag_to_index

# words in corpus /training examples
N_DATA = 719530

# Number of epochs to train the net
NUM_EPOCHS = 10

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 1

NUM_UNITS = 1

NUM_OUTPUT_UNITS = 11


def build_network(X_train, unique_tags, unique_tokens, W, longest_sent, input_var=None):

    print("Building network ...")

    l_in = lasagne.layers.InputLayer(shape=X_train.shape, input_var=input_var)
    print X_train.shape
    l_em = lasagne.layers.EmbeddingLayer(l_in, input_size=len(unique_tokens), output_size=100, W=W)
    l_re = lasagne.layers.RecurrentLayer(l_em, N_HIDDEN)
    l_out = lasagne.layers.DenseLayer(l_re, len(unique_tags), nonlinearity=lasagne.nonlinearities.softmax)
    return l_out

def main():

    input_var = T.lmatrix('inputs')
    target_var = T.dmatrix('targets')

    sents, tags, unique_tokens, unique_tags, longest_sent = prepare_sents(train_corpus)

    token_mappings = create_token_mappings(unique_tokens)
    tag_mappings = create_tag_mappings(unique_tags)


    X_train = np.asarray([[token_mappings[w] for w in sent] for sent in pad_sent(sents, longest_sent)])
    y_train = np.asarray([[tag_mappings[t] for t in sent_tags] for sent_tags in tags])


    # gensim 2D NumPy matrix
    model = gensim.models.Word2Vec(sents, min_count=1)
    # shape = (len(unique_tokens), 100)
    W = model.syn0.astype("float64")
    # print W.shape

    for epoch in range(NUM_EPOCHS):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 2, shuffle=True):
            print "q"
            inputs, targets = batch
            print inputs


    # for sent in X_train:
        network = build_network(X_train, unique_tags, unique_tokens, W, longest_sent, input_var)

        prediction = lasagne.layers.get_output(network)

        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

        train_fn = theano.function([input_var, target_var], loss, updates=updates)


        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])



    for epoch in range(NUM_EPOCHS):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train[:N_BATCH], y_train[:N_BATCH], 2, shuffle=True):
            inputs, targets = batch

            print inputs


            # err, acc = val_fn(inputs, targets)
            # train_err_ff += err

            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0

        X_val, y_val, unique_tags = prepare_data(val_corpus)
        # print len(X_val)
        for i in iterate_minibatches(X_val[:N_BATCH], y_val[:N_BATCH], 2, shuffle=False):
            inputs, targets = i
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.9f}s".format(epoch + 1, NUM_EPOCHS, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

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



if __name__ == '__main__':
    print main()

