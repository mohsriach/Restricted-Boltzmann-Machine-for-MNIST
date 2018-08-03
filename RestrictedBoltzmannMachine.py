#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# tile_raster_images function 
"""
For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""

import numpy
def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array
# End of Tile_raster_images function  
    
""" This is the main function | Restricted Boltzmann Machines """

import tensorflow as tf
import numpy as np
import PIL.Image as Img
#import utils
#from utils import tile_raster_images

# function for sampling probability
def sample_prob(probs):
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

# Hyperparameters
alpha = 0.01
batchsize = 100
epochs = 10

# importing the dataset 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# training and test data
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

# creating a placeholder
X = tf.placeholder("float", [None, 784])
# placeholder for weights and biases
rbm_w = tf.placeholder("float", [784, 500])
rbm_vb = tf.placeholder("float", [784])
rbm_hb = tf.placeholder("float", [500])

h0 = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v1 = sample_prob(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_w)) + rbm_vb))
h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_w) + rbm_hb)
# positive and negative gradients
w_positive_grad = tf.matmul(tf.transpose(X), h0)
w_negative_grad = tf.matmul(tf.transpose(v1), h1)
# update weights and bisases
update_w = rbm_w + alpha * (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(X)[0])
update_vb = rbm_vb + alpha * tf.reduce_mean(X - v1, 0)
update_hb = rbm_hb + alpha * tf.reduce_mean(h0 - h1, 0)
# perform gibbs sampling 
h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(X, rbm_w) + rbm_hb))
v_sample = sample_prob(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_w)) + rbm_vb))
# calculate reconstruction error
err = X - v_sample
err_sum = tf.reduce_mean(err * err)
# begin session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# initialise new and old weights and biases to zero 
n_w = np.zeros([784, 500], np.float32)
n_vb = np.zeros([784], np.float32)
n_hb = np.zeros([500], np.float32)
o_w = np.zeros([784, 500], np.float32)
o_vb = np.zeros([784], np.float32)
o_hb = np.zeros([500], np.float32)

print (sess.run(err_sum, feed_dict={X: trX, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb}))
# number of epochs
for i in range (0,epochs):
    # Number of Batches
    for start, end in zip(range(0, len(trX), batchsize), range(batchsize, len(trX), batchsize)):
        
        batch = trX[start:end]
        n_w = sess.run(update_w, feed_dict={
                   X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
        n_vb = sess.run(update_vb, feed_dict={
                    X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
        n_hb = sess.run(update_hb, feed_dict={
                    X: batch, rbm_w: o_w, rbm_vb: o_vb, rbm_hb: o_hb})
        o_w = n_w
        o_vb = n_vb
        o_hb = n_hb
        if start % 1000 == 0:
            # for every 1000 images
            print (sess.run(
            err_sum, feed_dict={X: trX, rbm_w: n_w, rbm_vb: n_vb, rbm_hb: n_hb}))
            image = Img.fromarray(
                    tile_raster_images(
                            X=n_w.T,
                            img_shape=(28, 28),
                            tile_shape=(25, 20),
                            tile_spacing=(1, 1)
                            )
                    )
            image.save("reconstructed_image_%d.png" % (start / 10000))
            # save images after every 10,000 trials