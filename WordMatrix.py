#!/usr/bin/env python
# -*- coding: utf-8 -*-

import string
import numpy as np

word_vectors = dict()
char_vocab = unicode(string.ascii_letters + string.punctuation + string.digits+"äöüÄÖÜß", "utf-8") #define vocabulary for characters


def init_dictionary():

    for char in char_vocab:
        generate_random_char_embedding(char)


def generate_random_char_embedding(char):

    is_unique = False

    if not word_vectors:
        #random_vector = np.random.rand(char_vocab.__len__(), 1)
        random_vector = np.random.uniform(-100, 100, (char_vocab.__len__(), 1))
        word_vectors[char] = random_vector
        is_unique = True

    while not is_unique:

        random_vector = np.random.uniform(-100, 100, (char_vocab.__len__(), 1))

        for key in word_vectors.keys():

            equal = np.array_equal(random_vector, word_vectors[key])

            if equal:
                is_unique = False
                break

            is_unique = not equal
            
        if is_unique:
            word_vectors[char] = random_vector


def generate_word_embedding(word):
    word = unicode(word ,"utf-8")
    word_embedding = None
    for char in word:
        print char
        if char in word_vectors:

            if word_embedding is not None:
                word_embedding = np.concatenate((word_embedding,word_vectors[char]),1)
            else:
                word_embedding = word_vectors[char]

    return word_embedding









#if __name__ == "__main__":
    #init_dictionary()

    #print generate_word_embedding("Hä")




