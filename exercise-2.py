#!/usr/bin/env python

import tensorflow as tf
import numpy as np

world = np.array([
    [0,0,0,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,1,0,0],
    [0,0,0,0,0]])

session = tf.Session()

maybe = []

for i in range(1, 4):
    for j in range(1, 4):
        neighbor_count_plus_self = tf.cast(
                tf.reduce_sum(
                    tf.strided_slice(world, begin=[i-1, j-1], end=[i+2, j+2], strides=[1, 1])), 
                    tf.int32
                )
        neighbor_count = tf.cond(tf.equal(1, world[i][j]), lambda: tf.subtract(neighbor_count_plus_self, 1), lambda: neighbor_count_plus_self)
        liveness = tf.cond(
                tf.equal(3, neighbor_count),
                lambda: 1,
                lambda: tf.cond(
                    tf.logical_and(
                        tf.equal(2, neighbor_count),
                        tf.equal(1, world[i][j]
                            )
                        ),
                    lambda: 1,
                    lambda: 0
                    )
                )
        maybe.append(liveness)
        #print session.run(liveness)

a = tf.reshape(maybe, [3,3])

print session.run(a)

