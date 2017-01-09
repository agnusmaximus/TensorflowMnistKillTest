from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Timer

from datetime import datetime
import os.path
import shutil
import time

import numpy as np
import random
import tensorflow as tf

import signal
import sys
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline

from tensorflow.python.ops import data_flow_ops

import mnist_data
import mnist
from sync_replicas_optimizer_modified import TimeoutReplicasOptimizer

# (rm -rf MNISTKillTest.py_logdir) && (python MNISTKillTest.py 0 ps > /tmp/output2 2>&1 &) && (python MNISTKillTest.py 0 worker)

index, job = int(sys.argv[1]), sys.argv[2]
ps_hosts = 'localhost:1234'
worker_hosts = 'localhost:1235'
ps_hosts = ps_hosts.split(",")
worker_hosts = worker_hosts.split(",")
cluster_spec = tf.train.ClusterSpec({'ps' : ps_hosts, 'worker' : worker_hosts})
server = tf.train.Server(
    {'ps': ps_hosts,
     'worker': worker_hosts},
    job_name=job,
    task_index=index)


# ---------------------------------------------
# Model setup
# ---------------------------------------------

dataset = mnist_data.load_mnist(worker_id=job, n_workers=1).train

# Ops are assigned to worker by default.
with tf.device(
    tf.train.replica_device_setter(
      worker_device='/job:worker/task:%s' % index,
      cluster=cluster_spec)):

  # Create a variable to count the number of train() calls. This equals the
  # number of updates applied to the variables. The PS holds the global step.
  global_step = tf.Variable(0, name="global_step", trainable=False)

  # Calculate the learning rate schedule.
  num_batches_per_epoch = (dataset.num_examples / 128)

  # Decay steps need to be divided by the number of replicas to aggregate.
  # This was the old decay schedule. Don't want this since it decays too fast with a fixed learning rate.
  decay_steps = int(num_batches_per_epoch)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(.001,
                                  global_step,
                                  decay_steps,
                                  1,
                                  staircase=True)

  images, labels = mnist.placeholder_inputs(128)

  # Number of classes in the Dataset label set plus 1.
  # Label 0 is reserved for an (unused) background class.
  logits = mnist.inference(images, train=True)

  # Add classification loss.
  total_loss = mnist.loss(logits, labels)

  # Create an optimizer that performs gradient descent.
  opt = tf.train.AdamOptimizer(lr)

  opt = TimeoutReplicasOptimizer(
      opt,
      global_step,
      total_num_replicas=1)

  # Compute gradients with respect to the loss.
  grads = opt.compute_gradients(total_loss)
  apply_gradients_op = opt.apply_gradients(grads, index, global_step=global_step, collect_cdfs=False)

  with tf.control_dependencies([apply_gradients_op]):
    train_op = tf.identity(total_loss, name='train_op')

# -----------------------------------------------
# Distributed setup
# -----------------------------------------------
if os.path.exists('./test_logdir'):
    shutil.rmtree('./test_logdir')

sv = tf.train.Supervisor(is_chief=True, logdir='./test_logdir')
cfg = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False)
sess = sv.prepare_or_wait_for_session(server.target, config=cfg)

print("Testing timeout...")
print("--------------------------------")

def interval_updates():
    sess.run([opt._update_op])
    Timer(5, kill_session).start()

def kill_session():
    sess.kill()
    Timer(7, kill_session).start()

Timer(7, kill_session).start()
Timer(5, interval_updates).start()

while True:
    try:

        feed_dict = mnist.fill_feed_dict(dataset, images, labels, 128)
        loss = sess.run([train_op], feed_dict=feed_dict)
        printout_str = "Loss: %f" % loss[0]
        print(printout_str)

    except tf.errors.DeadlineExceededError:
        print("Successfully timed out!")
        print("Update op...")
