#!/usr/bin/env python

import tensorflow as tf
import numpy as np

session = tf.Session()

def gol_loss(world_next_probabilities, world_next_target):
  """Calculates loss between 2D Game of Life predictions and targets.
  For a reference on Game of Life, see
  [Wikipedia page](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).
  The implementation assumes:
   * 2D square world of fixed size.
   * Edge cells are always dead.
  Args:
    world_next_probabilities: A rank-2 `Tensor` representing probabilities
      that each cell lives.
    targets: A rank-2 `Tensor` representing the targets, that is, whether the
      cell actually lives or not.
  Returns:
    A `float`, the loss value.
  """
  with tf.name_scope("gol_loss"):
    return tf.contrib.losses.log_loss(world_next_probabilities, world_next_target)

def generate_gol_example(size):
  """Generates a random pair of Game of Life 2D world and its next time step.

  For a reference on Game of Life, see
  [Wikipedia page](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).

  The implementation assumes:
   * 2D square world of fixed size.
   * Edge cells are always dead.

  Args:
    size: An `int` indicating the size of the world. Must be at least 3.

  Returns:
    A tuple `(world, next_world)`:
     * world: A `Tensor` of shape `(size, size)`, representing a random GoL
       world.
     * world_next: A `Tensor` of shape `(size, size)`, representing `world`
       after one time step.

  """
  if size < 3:
    raise ValueError("Size must be greater than 2, received %d" % size)

  with tf.name_scope("generate_gol_example"):
    world = tf.random_uniform(
        (size-2, size-2), minval=0, maxval=2, dtype=tf.int32)
    world_padded = tf.pad(world, [[1, 1], [1, 1]])

    num_neighbors = (
        world_padded[:-2, :-2]
        + world_padded[:-2, 1:-1]
        + world_padded[:-2, 2:]

        + world_padded[1:-1, :-2]
        + world_padded[1:-1, 2:]

        + world_padded[2:, :-2]
        + world_padded[2:, 1:-1]
        + world_padded[2:, 2:]
    )

    cell_survives = tf.logical_or(
        tf.equal(num_neighbors, 3), tf.equal(num_neighbors, 2))
    cell_rebirths = tf.equal(num_neighbors, 3)

    survivors = tf.where(
        cell_survives, world_padded[1:-1, 1:-1], tf.zeros_like(world))
    world_next = tf.where(cell_rebirths, tf.ones_like(world), survivors)

    world_next_padded = tf.pad(world_next, [[1, 1], [1, 1]])

    return world_padded, world_next_padded

def train_model(size):
    # cheat and generate correct data
    # give input / output to optimizer
    world, world_next = generate_gol_example(size)
    flat_world = tf.to_float(tf.reshape(world, [1, -1]))
    flat_world_next = tf.to_float(tf.reshape(world_next, [1, -1]))
    hidden_layer = tf.contrib.layers.fully_connected(
            inputs = flat_world,
            num_outputs = (size * size))
    prediction = tf.reshape(
            tf.to_int32(
                tf.greater(hidden_layer, 0.5)),
            [size, size])
    loss = gol_loss(hidden_layer, flat_world_next)
    optimizer = tf.train.AdagradOptimizer(1.0)
    return world, world_next, loss, prediction, optimizer.minimize(loss)

w, wn, l, p, o = train_model(5)
