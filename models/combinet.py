from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization

def create_network(img_prep, img_aug, learning_rate):
    """This function defines the network structure.

    Args:
        img_prep: Preprocessing function that will be done to each input image.
        img_aug: Data augmentation function that will be done to each training input image.

    Returns:
        The network."""

    # Input shape will be [batch_size, height, width, channels].
    network = input_data(shape=[None, 64, 64, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    # First convolution layer. 32 filters of size 5. Activation function ReLU. 64x64x3 -> 64x64x32
    network = conv_2d(network, 32, 5, activation='relu')
    # Pooling layer. 64x64x32 -> 32x32x32
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 32, 5, activation='relu')
    # Pooling layer. 64x64x32 -> 32x32x32
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 32, 5, activation='relu')
    # Pooling layer. 64x64x32 -> 32x32x32
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 32, 5, activation='relu')
    # Pooling layer. 64x64x32 -> 32x32x32
    network = max_pool_2d(network, 2)

    network = conv_2d(network, 32, 5, activation='relu')
    network = conv_2d(network, 32, 5, activation='relu')
    # Pooling layer. 64x64x32 -> 32x32x32
    network = max_pool_2d(network, 2)

    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # First fully connected layer. 32x32x32 -> 1x32768 -> 1x1024. ReLU activation.
    network = fully_connected(network, 1024, activation='relu')
    
    network = batch_normalization(network, stddev=0.002, trainable=True, restore=True, reuse=False)
    # Dropout layer for the first fully connected layer.
    network = dropout(network, 0.5)
    
    network = fully_connected(network, 200, activation='softmax')
    
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=learning_rate)
    return network