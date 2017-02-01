#!/usr/bin/env python
# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import lasagne as layered_noodle_cake
import lasagne.layers as L
import numpy as np
import string
import WordMatrix

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
    data_size = (None,1,None)  # Batch size x Img Channels x Height x Width



    input_var = T.imatrix('input')
    input_var = T.tensor3(name = "input",dtype='int32')


    input_layer = L.InputLayer(data_size, input_var=input_var)

    W = create_char_embedding_matrix()

    embed_layer = L.EmbeddingLayer(input_layer, input_size=102, output_size=101, W=W)

    conv_layer_1 = L.Conv2DLayer(embed_layer, 4, 1, 1, 0)
    pool_layer_1 = L.Pool2DLayer(conv_layer_1, pool_size=1)

    conv_layer_2 = L.Conv2DLayer(embed_layer, 4, 2, 1, 1)
    pool_layer_2 = L.Pool2DLayer(conv_layer_2, pool_size=2)

    conv_layer_3 = L.Conv2DLayer(embed_layer, 4, 3, 1, 2)
    pool_layer_3 = L.Pool2DLayer(conv_layer_3, pool_size=3)

    merge_layer = L.ConcatLayer([pool_layer_1, pool_layer_2, pool_layer_3], 0)



    target_var = T.ivector('targets')

    output = L.get_output(merge_layer,T.matrix(name = "input",dtype='int32'))

    f = theano.function([T.matrix(name = "input",dtype='int32')],output)
    x_test = np.array([[0, 2,3]]).astype('int32')
    print f(x_test)
    return merge_layer


if __name__ == "__main__":
    #testing()

    cnn = build_cnn()

    #x = T.tensor4(name = "input",dtype='int32')


    #
    #print L.get_output(cnn)
    #create_char_embedding_matrix()
