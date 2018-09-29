#!/usr/bin/python

#===============================================
# image_manip.py
#
# some helpful hints for those of you
# who'll do the final project in Py
#
# bugs to vladimir dot kulyukin at usu dot edu
#===============================================

import cv2
import os
import network
import numpy as np

def load_data():

    TRAIN_BEE = []
    TRAIN_NO_BEE = []

    TRAIN_TARGET_BEE = []
    TRAIN_TARGET_NO_BEE = []

    TEST_BEE = []
    TEST_NO_BEE = []

    TEST_TARGET_BEE = []
    TEST_TARGET_NO_BEE = []

    ROOT_DIR = './nn_train/'

    ## read the single bee train images
    YES_BEE_TRAIN = ROOT_DIR + 'single_bee_train'
    for root, dirs, files in os.walk(YES_BEE_TRAIN):
        for item in files:
            if item.endswith('.png'):
                ip = os.path.join(root, item)
                img = cv2.imread(ip)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = (img.flatten()/float(255))
                TRAIN_BEE.append(list(img))
                TRAIN_TARGET_BEE.append(int(1))

    TRAIN_BEE = np.array(TRAIN_BEE)
    TRAIN_TARGET_BEE = np.array(TRAIN_TARGET_BEE)
    TRAIN_IMAGE_BEE = (TRAIN_BEE, TRAIN_TARGET_BEE)
    #print TRAIN_IMAGE_BEE

    ## read the no-bee train images
    NO_BEE_TRAIN = ROOT_DIR + 'no_bee_train'
    for root, dirs, files in os.walk(NO_BEE_TRAIN):
        for item in files:
            if item.endswith('.png'):
                ip = os.path.join(root, item)
                img = cv2.imread(ip)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = (img.flatten()/float(255))
                TRAIN_NO_BEE.append(list(img))
                TRAIN_TARGET_NO_BEE.append(int(0))

    TRAIN_NO_BEE = np.array(TRAIN_NO_BEE)
    TRAIN_TARGET_NO_BEE = np.array(TRAIN_TARGET_NO_BEE)
    TRAIN_IMAGE_NO_BEE = (TRAIN_NO_BEE, TRAIN_TARGET_NO_BEE)

    #comninate the train image with bee and no-bee
    TRAIN_IMAGE_DATA = np.concatenate((TRAIN_BEE, TRAIN_NO_BEE))
    TRAIN_TARGET = np.concatenate((TRAIN_TARGET_BEE, TRAIN_TARGET_NO_BEE))
    TRAIN_IMAGE_CLASSIFICATIONS = (TRAIN_IMAGE_DATA, TRAIN_TARGET)
    #print TRAIN_IMAGE_CLASSIFICATIONS


    ## read the single bee test images
    YES_BEE_TEST = ROOT_DIR + 'single_bee_test'
    for root, dirs, files in os.walk(YES_BEE_TEST):
        for item in files:
            if item.endswith('.png'):
                ip = os.path.join(root, item)
                img = cv2.imread(ip)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = (img.flatten()/float(255))
                TEST_BEE.append(list(img))
                TEST_TARGET_BEE.append(int(1))

    TEST_BEE = np.array(TEST_BEE)
    TEST_TARGET_BEE = np.array(TEST_TARGET_BEE)
    TEST_IMAGE_BEE = (TEST_BEE, TEST_TARGET_BEE)



    # read the no-bee test images
    NO_BEE_TEST = ROOT_DIR + 'no_bee_test'
    for root, dirs, files in os.walk(NO_BEE_TEST):
        for item in files:
            if item.endswith('.png'):
                ip = os.path.join(root, item)
                img = cv2.imread(ip)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = (img.flatten()/float(255))
                TEST_NO_BEE.append(list(img))
                TEST_TARGET_NO_BEE.append(int(0))

    TEST_NO_BEE = np.array(TEST_NO_BEE)
    TEST_TARGET_NO_BEE = np.array(TEST_TARGET_NO_BEE)
    TEST_IMAGE_NO_BEE = (TEST_NO_BEE, TEST_TARGET_NO_BEE)


    TEST_IMAGE_DATA = np.concatenate((TEST_BEE, TEST_NO_BEE))
    TEST_TARGET = np.concatenate((TEST_TARGET_BEE, TEST_TARGET_NO_BEE))
    TEST_IMAGE_CLASSIFICATIONS = (TEST_IMAGE_DATA, TEST_TARGET)
    #print TEST_IMAGE_CLASSIFICATIONS

    return (TRAIN_IMAGE_BEE, TRAIN_IMAGE_NO_BEE, TEST_IMAGE_BEE, TEST_IMAGE_NO_BEE, TRAIN_IMAGE_CLASSIFICATIONS, TEST_IMAGE_CLASSIFICATIONS)


def load_data_wrapper():
    tr_bee, tr_no_bee, te_bee, te_no_bee, tr_d, te_d = load_data()

    training_bee_inputs = [np.reshape(x, (1024, 1)) for x in tr_bee[0]]
    training_bee_results = [vectorized_result(y) for y in tr_bee[1]]
    training_bee_data = zip(training_bee_inputs, training_bee_results)

    training_no_bee_inputs = [np.reshape(x, (1024, 1)) for x in tr_no_bee[0]]
    training_no_bee_results = [vectorized_result(y) for y in tr_no_bee[1]]
    training_no_bee_data = zip(training_no_bee_inputs, training_no_bee_results)

    training_inputs = [np.reshape(x, (1024, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    #print training_data

    test_bee_inputs = [np.reshape(x, (1024, 1)) for x in te_bee[0]]
    test_bee_results = [vectorized_result(y) for y in te_bee[1]]
    test_bee_data = zip(test_bee_inputs, test_bee_results)

    test_no_bee_inputs = [np.reshape(x, (1024, 1)) for x in te_no_bee[0]]
    test_no_bee_results = [vectorized_result(y) for y in te_no_bee[1]]
    test_no_bee_data = zip(test_no_bee_inputs, test_no_bee_results)


    test_inputs = [np.reshape(x, (1024, 1)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    #print test_data


    return (training_bee_data, training_no_bee_data, test_bee_data, test_no_bee_data, training_data, test_data)

def vectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

#t1,t2 = load_data_wrapper()
load_data_wrapper()























