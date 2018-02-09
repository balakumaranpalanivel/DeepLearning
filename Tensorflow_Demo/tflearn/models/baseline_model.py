"""Tutorial for the CS7GV1 Computer Vision 17/18 lecture at Trinity College Dublin.

This script gives the network definition."""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def create_network(img_prep, img_aug, learning_rate):
    """This function defines the network structure.

    Args:
        img_prep: Preprocessing function that will be done to each input image.
        img_aug: Data augmentation function that will be done to each training input image.

    Returns:
        The network."""

    # Input shape will be [batch_size, height, width, channels].
    network = input_data(shape=[None, 56, 56, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
                         
    # Layer #1 - 1st convolution layer.
    network = conv_2d(network, 56, 56, activation='relu')

    # Max Pool Layer
    network = max_pool_2d(network, 3, strides=2)

    # Local Response Normalisation layer
    network = local_response_normalization(network)

    # Layer #2 - 2nd convolution layer
    network = conv_2d(network, 28, 28, activation='relu')

    # Max Pool Layer
    network = max_pool_2d(network, 3, strides=2)

    # Local Response Normalisation Layer
    network = local_response_normalization(network)

    # Layer #3 - 3rd convolution layer
    network = conv_2d(network, 14, 14, activation='relu')

    # Layer #4 - 4th convolution layer
    network = conv_2d(network, 14, 14, activation='relu')

    # Layer #5 - 5th convolution layer
    network = conv_2d(network, 14, 14, activation='relu')

    # Max Pool layer
    network = max_pool_2d(network, 3, strides=2)

    # Local Response Normalisation Layer
    network = local_response_normalization(network)

    # Layer #6 - Fully Connected layer
    network = fully_connected(network, 1024, activation='tanh')

    # Dropout
    network = dropout(network, 0.5)

    # Layer #7 - Fully Connected layer
    network = fully_connected(network, 1024, activation='tanh')

    # Dropout
    network = dropout(network, 0.5)

    # Layer #8 - Fully Connected layer 
    network = fully_connected(network, 200, activation='softmax')

    network = regression(network, optimizer='momentum',
                        loss='categorical_crossentropy',
                        learning_rate=learning_rate)

    return network