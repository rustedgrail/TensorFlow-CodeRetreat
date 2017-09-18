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

maybe = tf.zeros([3,3])

for i in range(0, 3):
    for j in range(0, 3):
        if i == 1 and j == 1:
            continue
        neighbor_chunk = tf.strided_slice(
                world,
                begin=[i, j],
                end=[i+3, j+3],
                strides=[1,1]
                )
        maybe = tf.add(maybe, neighbor_chunk)

print session.run(maybe)
