#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import lasagne as layered_noodle_cake
import lasagne.layers as L
import numpy as np
import string
import WordMatrix
theano.config.compute_test_value = "warn"
char_vocab = unicode(string.ascii_letters + string.punctuation + string.digits+"äöüÄÖÜß", "utf-8") #define vocabulary for characters
char_index_lookup = dict()

def create_char_embedding_matrix():
    index = 0
    embedding_matrix = WordMatrix.generate_random_char_embedding("unknown")
    char_index_lookup["unknown"] = index

    for char in char_vocab:
        index += 1
        new_vector = WordMatrix.generate_random_char_embedding(char)

        embedding_matrix = np.concatenate((embedding_matrix, new_vector), 0)
        char_index_lookup[char] = index

    return embedding_matrix

def build_cnn():
    data_size = (None,10,100)  # Batch size x Img Channels x Height x Width

    input_var = T.tensor3(name = "input",dtype='int64')

    values = np.array(np.random.randint(0,1,(5,10,100)))
    input_var.tag.test_value = values
    input_layer = L.InputLayer(data_size, input_var=input_var)

    W = create_char_embedding_matrix()

    embed_layer = L.EmbeddingLayer(input_layer, input_size=102,output_size=101, W=W)


    reshape = L.reshape(embed_layer,(-1,100,101))
    dim_shuffle = L.dimshuffle(reshape,(0,2,1))
    #conv_layer_1 = L.Conv2DLayer(embed_layer, 4, (1), 1, 0)
    #pool_layer_1 = L.MaxPool1DLayer(conv_layer_1, pool_size=1)
    print L.get_output(dim_shuffle).tag.test_value.shape

    conv_layer_1 = L.Conv1DLayer(dim_shuffle, 50, 2, 1)


    print L.get_output(conv_layer_1).tag.test_value.shape
    print "TEST"
    pool_layer_1 = L.MaxPool1DLayer(conv_layer_1, pool_size=99)
    print L.get_output(pool_layer_1).tag.test_value.shape
    reshape_conv_1 = L.reshape(pool_layer_1,(-1,50))

    conv_layer_2 = L.Conv1DLayer(dim_shuffle, 50, 3, 1)
    pool_layer_2 = L.MaxPool1DLayer(conv_layer_2, pool_size=98)
    reshape_conv_2 = L.reshape(pool_layer_2, (-1, 50))

    merge_layer = L.ConcatLayer([reshape_conv_1, reshape_conv_2], 1)
    print L.get_output(merge_layer).tag.test_value.shape
    reshape_output = L.reshape(merge_layer,(-1,10,100))
    print L.get_output(reshape_output).tag.test_value.shape

    x = T.tensor3(name = "testname",dtype='int32')
    #x = T.imatrix()
    #output = L.get_output(conv_layer_1,x)

    #f = theano.function([x],output)

    word = unicode("Tat")
    word_index  = np.array([])

    #print word_index

    #x_test = np.array([word_index]).astype('int32')
    #print f(x_test)

    return reshape_output


if __name__ == "__main__":
    #testing()

    build_cnn()

    #x = T.tensor4(name = "input",dtype='int32')


    #
    #print L.get_output(cnn)
    #create_char_embedding_matrix()
