#!/usr/bin/env python

import tensorflow as tf
import numpy as np

world = np.array([
    [0,0,0,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,0,0,0]], dtype='float32')

session = tf.Session()

prepped_for_convolution_world = tf.reshape(world, [1,5,5,1])
filter = tf.constant(
        np.array([
            [1,1,1],
            [1,0,1],
            [1,1,1]]),
        dtype=tf.float32)

filter = tf.reshape(filter, [3,3,1,1])

conv = tf.nn.conv2d(prepped_for_convolution_world, filter, [1,1,1,1], "VALID")

neighbor_map = tf.reshape(conv, [3,3])
neighbor_map = tf.pad(neighbor_map, [[1,1],[1,1]])

dead_world = tf.zeros_like(neighbor_map)
living_world = tf.ones_like(neighbor_map)

portal_world = tf.where(tf.equal(3.0, neighbor_map), living_world, dead_world)
still_hanging_on = tf.where(tf.equal(2.0, neighbor_map), world, dead_world)

new_world = tf.add(portal_world, still_hanging_on)

print session.run(new_world)


