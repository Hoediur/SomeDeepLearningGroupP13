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
#import gensim

import CNN
import lasagne.layers as L

import WordMatrix
import string


from datetime import datetime
dt = datetime.now()



# ############################## Data preparation ################################

unknown_token = "UNKNOWN_TOKEN"
theano.config.compute_test_value = "ignore"
char_vocab = unicode(string.ascii_letters + string.punctuation + string.digits+"äöüÄÖÜß", "utf-8") #define vocabulary for characters
char_index_table = dict()


batch_size = 1

def create_char_embedding_matrix():
    index = 0
    embedding_matrix = WordMatrix.generate_random_char_embedding("unknown")
    char_index_table["unknown"] = index
    index += 1

    new_vector = WordMatrix.generate_random_char_embedding("padding")
    embedding_matrix = np.concatenate((embedding_matrix, new_vector), 0)

    char_index_table["padding"] = index


    for char in char_vocab:
        index += 1
        new_vector = WordMatrix.generate_random_char_embedding(char)

        embedding_matrix = np.concatenate((embedding_matrix, new_vector), 0)
        char_index_table[char] = index

    return embedding_matrix


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
    longest_word = 0
    with open(corpus) as f:
        for line in f:

            if line.strip(): # check if line not empty
                token = re.split(r'\t', line)[0]
                tag = re.split(r'\t', line)[1].strip()
                sent.append(token)
                sent_tags.append(tag)

                unique_tags.add(tag)
                unique_tokens.add(token)
                token_length = len(token)
                if token_length > longest_word:
                    longest_word = token_length
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
    print "longest word", longest_word
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
        tag_to_vector[tag] = np.zeros(len(unique_tags))
        tag_to_vector[tag][i] = 1
        #tag_to_vector[tag] = i
    return tag_to_index, tag_to_vector

# Number of epochs to train the net
NUM_EPOCHS = 500

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 10

NUM_UNITS = 10
max_sentence = 0



def build_network(W,number_unique_tags,longest_word,longest_sentence, input_var=None):
    print("Building network ...")

    input_layer = L.InputLayer((None,longest_sentence,longest_word), input_var=input_var)

    embed_layer = L.EmbeddingLayer(input_layer, input_size=103,output_size=101, W=W)

    reshape_embed = L.reshape(embed_layer,(-1,longest_word,101))

    conv_layer_1 = L.Conv1DLayer(reshape_embed, longest_word, 2)
    conv_layer_2 = L.Conv1DLayer(reshape_embed, longest_word, 3)

    pool_layer_1 = L.MaxPool1DLayer(conv_layer_1, pool_size=longest_word-1)
    pool_layer_2 = L.MaxPool1DLayer(conv_layer_2, pool_size=longest_word-2)

    merge_layer = L.ConcatLayer([pool_layer_1, pool_layer_2], 1)
    flatten_merge = L.flatten(merge_layer, 2)
    reshape_merge = L.reshape(flatten_merge, (-1,longest_sentence,int(longest_word*2)))

    l_re = lasagne.layers.RecurrentLayer(reshape_merge, N_HIDDEN, nonlinearity=lasagne.nonlinearities.sigmoid, mask_input=None)
    l_out = lasagne.layers.DenseLayer(l_re, number_unique_tags, nonlinearity=lasagne.nonlinearities.softmax)

    print "DONE BUILDING NETWORK"
    return l_out



# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. This function returns only mini-batches
# of size `batchsize`. If the size of the data is not a multiple of `batchsize`,
# it will not return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize=1, shuffle=False):

    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]




def create_word_index_vectors(corpus,max_length):
    result = None
    n = 0
    print len(corpus)
    for sent in corpus:
        temp_result = None

        for word in sent:

            n +=1
            if n % 10000 == 0:
                print n
            word_array = []

            for char in unicode(word,"utf-8"):

                if char in char_index_table:
                    word_array.append(char_index_table[char])
                else:
                    word_array.append(char_index_table["unknown"])

            delta_word = max_length - len(word_array)

            if delta_word >0:
                word_array += [char_index_table["padding"]]*delta_word

            if temp_result is None:
                temp_result = np.array([word_array])
            else:
                temp_result = np.append(temp_result, [np.array(word_array)], axis=0)
        if result is None:
            result = np.array([temp_result])
        else:
            result = np.append(result,[temp_result],axis=0)
        return result

def create_tag_vectors(corpus_tag_sequence,tag_vector_mappings):
    result = None
    result_test = []
    n = 0
    for tags in corpus_tag_sequence:

        for tag in tags:
            n += 1
            if n % 10000 == 0:
                print n

            if not result_test:
                result_test = [tag_vector_mappings[tag].tolist()]
            else:
                result_test.append(tag_vector_mappings[tag].tolist())

    result = np.array(result_test)
    print result
    return result
# ############################## Main program ################################

def main():
    max_word_length = 55 #magicNumber

    input_var = T.tensor3(name="input", dtype='int64')
    target_var = T.dmatrix('targets')

    W = create_char_embedding_matrix()

    # Load the dataset
    print "Loading data..."

    sents_train, tags_train, unique_tokens_train, unique_tags_train, longest_sent_train = prepare_sents(train_corpus)

    token_mappings = create_token_mappings(unique_tokens_train)
    tag_index_mappings, tag_vector_mappings = create_tag_mappings(unique_tags_train)

    pad_sent(sents_train, tags_train, longest_sent_train)

    X_train = create_word_index_vectors(sents_train,max_word_length)
    y_train = create_tag_vectors(tags_train,tag_vector_mappings)

    sents_val, tags_val, unique_tokens_val, unique_tags_val, longest_sent_val = prepare_sents(val_corpus)

    pad_sent(sents_val, tags_val, longest_sent_train)
    X_val = create_word_index_vectors(sents_val, max_word_length)
    y_val = create_tag_vectors(tags_val, tag_vector_mappings)

    sents_test, tags_test, unique_tokens_test, unique_tags_test, longest_sent_test = prepare_sents(test_corpus)

    pad_sent(sents_test, tags_test, longest_sent_train)
    X_test = create_word_index_vectors(sents_test, max_word_length)
    y_test = create_tag_vectors(tags_test, tag_vector_mappings)

    network = build_network(W,len(unique_tags_train),max_word_length,int(max(X_train.shape[1],X_test.shape[1],X_val.shape[1])), input_var)

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
        loss, params, learning_rate=0.0001, momentum=0.9)

    #updates = lasagne.updates.sgd(loss,params,learning_rate=0.0001)
    #updates = lasagne.updates.adadelta(loss,params, learning_rate=0.0001)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    # we also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=0), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    print y_train.shape
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    print "training"
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(NUM_EPOCHS):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            if train_batches % 1000 == 0:
                print train_batches

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
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
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        print acc
        test_acc += acc
        test_batches += 1
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

if __name__ == '__main__':
    train_corpus = "de-train.txt"
    val_corpus = "de-dev.txt"
    test_corpus = "de-test.txt"
    main()




