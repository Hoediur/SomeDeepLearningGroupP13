#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import lasagne as layered_noodle_cake
import lasagne.layers as L
import numpy as np
import string
import WordMatrix
theano.config.compute_test_value = "ignore"
char_vocab = unicode(string.ascii_letters + string.punctuation + string.digits+"äöüÄÖÜß", "utf-8") #define vocabulary for characters
char_index_lookup = dict()

def create_char_embedding_matrix():
    index = 0
    embedding_matrix = WordMatrix.generate_random_char_embedding("unknown")
    char_index_lookup["unknown"] = index
    index += 1

    new_vector = WordMatrix.generate_random_char_embedding("padding")
    embedding_matrix = np.concatenate((embedding_matrix, new_vector), 0)

    char_index_lookup["padding"] = index


    for char in char_vocab:
        index += 1
        new_vector = WordMatrix.generate_random_char_embedding(char)

        embedding_matrix = np.concatenate((embedding_matrix, new_vector), 0)
        char_index_lookup[char] = index

    return embedding_matrix

def build_cnn(input):
    #data_size = (None,103,130)  # Batch size x Img Channels x Height x Width

    #input_var = T.tensor3(name = "input",dtype='int64')
    input_var = input

    #values = np.array(np.random.randint(0,102,(1,9,50)))

    #input_var.tag.test_value = values
    #number sentences x words x characters
    input_layer = L.InputLayer((None,9,50), input_var=input)

    W = create_char_embedding_matrix()

    embed_layer = L.EmbeddingLayer(input_layer, input_size=103,output_size=101, W=W)
    #print "EMBED", L.get_output(embed_layer).tag.test_value.shape
    reshape_embed = L.reshape(embed_layer,(-1,50,101))
    #print "reshap embed", L.get_output(reshape_embed).tag.test_value.shape
    conv_layer_1 = L.Conv1DLayer(reshape_embed, 55, 2)
    conv_layer_2 = L.Conv1DLayer(reshape_embed, 55, 3)
    #print "TEST"
    #print "Convolution Layer 1", L.get_output(conv_layer_1).tag.test_value.shape
    #print "Convolution Layer 2", L.get_output(conv_layer_2).tag.test_value.shape

    #flatten_conv_1 = L.flatten(conv_layer_1,3)
    #flatten_conv_2 = L.flatten(conv_layer_2,3)

    #reshape_max_1 = L.reshape(flatten_conv_1,(-1,49))
    #reshape_max_2 = L.reshape(flatten_conv_2, (-1,48))

    #print "OUTPUT Flatten1", L.get_output(flatten_conv_1).tag.test_value.shape
    #print "OUTPUT Flatten2", L.get_output(flatten_conv_2).tag.test_value.shape

    #print "OUTPUT reshape_max_1", L.get_output(reshape_max_1).tag.test_value.shape
    #print "OUTPUT reshape_max_2", L.get_output(reshape_max_2).tag.test_value.shape

    pool_layer_1 = L.MaxPool1DLayer(conv_layer_1, pool_size=54)
    pool_layer_2 = L.MaxPool1DLayer(conv_layer_2, pool_size=53)


    #print "OUTPUT POOL1", L.get_output(pool_layer_1).tag.test_value.shape
    #print "OUTPUT POOL2",L.get_output(pool_layer_2).tag.test_value.shape

    merge_layer = L.ConcatLayer([pool_layer_1, pool_layer_2], 1)

    flatten_merge = L.flatten(merge_layer, 2)
    reshape_merge = L.reshape(flatten_merge, (1,9,110))
    print L.get_output(reshape_embed).shape
    #print L.get_output(reshape_merge).tag.test_value.shape

    return reshape_merge, char_index_lookup


if __name__ == "__main__":
    #testing()

    build_cnn(T.tensor3(name = "testname",dtype='int32'))

    #x = T.tensor4(name = "input",dtype='int32')


    #
    #print L.get_output(cnn)
    #create_char_embedding_matrix()
