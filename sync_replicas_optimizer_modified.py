# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Synchronize replicas for training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from threading import Timer
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import logging_ops

from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner

tf.app.flags.DEFINE_float('interval_ms', 1000, 'The interval ms')

FLAGS = tf.app.flags.FLAGS

# Please note that the gradients from replicas are averaged instead of summed
# (as in the old sync_replicas_optimizer) so you need to increase the learning
# rate according to the number of replicas. This change is introduced to be
# consistent with how gradients are aggregated (averaged) within a batch in a
# replica.
class TimeoutReplicasOptimizer(optimizer.Optimizer):
  """Class to synchronize, aggregate gradients and pass them to the optimizer.
  In a typical asynchronous training environment, it's common to have some
  stale gradients. For example, with a N-replica asynchronous training,
  gradients will be applied to the variables N times independently. Depending
  on each replica's training speed, some gradients might be calculated from
  copies of the variable from several steps back (N-1 steps on average). This
  optimizer avoids stale gradients by collecting gradients from all replicas,
  averaging them, then applying them to the variables in one shot, after
  which replicas can fetch the new variables and continue.
  The following accumulators/queue are created:
  <empty line>
  * N `gradient accumulators`, one per variable to train. Gradients are pushed
    to them and the chief worker will wait until enough gradients are collected
    and then average them before applying to variables. The accumulator will
    drop all stale gradients (more details in the accumulator op).
  * 1 `token` queue where the optimizer pushes the new global_step value after
    all variables are updated.
  The following local variable is created:
  * `sync_rep_local_step`, one per replica. Compared against the global_step in
    each accumulator to check for staleness of the gradients.
  The optimizer adds nodes to the graph to collect gradients and pause the
  trainers until variables are updated.
  For the Parameter Server job:
  <empty line>
  1. An accumulator is created for each variable, and each replica pushes the
     gradients into the accumulators instead of directly applying them to the
     variables.
  2. Each accumulator averages once enough gradients (replicas_to_aggregate)
     have been accumulated.
  3. Apply the averaged gradients to the variables.
  4. Only after all variables have been updated, increment the global step.
  5. Only after step 4, pushes `global_step` in the `token_queue`, once for
     each worker replica. The workers can now fetch the global step, use it to
     update its local_step variable and start the next batch.
  For the replicas:
  <empty line>
  1. Start a step: fetch variables and compute gradients.
  2. Once the gradients have been computed, push them into gradient
     accumulators. Each accumulator will check the staleness and drop the stale.
  3. After pushing all the gradients, dequeue an updated value of global_step
     from the token queue and record that step to its local_step variable. Note
     that this is effectively a barrier.
  4. Start the next batch.
  ### Usage
  ```python
  # Create any optimizer to update the variables, say a simple SGD:
  opt = GradientDescentOptimizer(learning_rate=0.1)
  # Wrap the optimizer with sync_replicas_optimizer with 50 replicas: at each
  # step the optimizer collects 50 gradients before applying to variables.
  # Note that if you want to have 2 backup replicas, you can change
  # total_num_replicas=52 and make sure this number matches how many physical
  # replicas you started in your job.
  opt = tf.SyncReplicasOptimizerV2(opt, replicas_to_aggregate=50,
                                   total_num_replicas=50)
  # Some models have startup_delays to help stabilize the model but when using
  # sync_replicas training, set it to 0.
  # Now you can call `minimize()` or `compute_gradients()` and
  # `apply_gradients()` normally
  grads = opt.minimize(total_loss, global_step=self.global_step)
  # You can now call get_init_tokens_op() and get_chief_queue_runner().
  # Note that get_init_tokens_op() must be called before creating session
  # because it modifies the graph by adding new nodes.
  init_token_op = opt.get_init_tokens_op()
  chief_queue_runner = opt.get_chief_queue_runner()
  ```
  In the training program, every worker will run the train_op as if not
  synchronized. But one worker (usually the chief) will need to execute the
  chief_queue_runner and get_init_tokens_op from this optimizer.
  ```python
  # When you create the supervisor, you need to add the local_init_op and
  # ready_for_local_init_op to make sure the local_step is initialized to the
  # global_step. Here is an example:
  if is_chief:
    local_init_op = opt.chief_init_op
  else:
    local_init_op = opt.local_step_init_op
  ready_for_local_init_op = opt.ready_for_local_init_op
  sv = tf.Supervisor(graph=g,
                     is_chief=is_chief,
                     # This initialize local step.
                     local_init_op=local_init_op,
                     # This makes sure global step is initialized before using.
                     ready_for_local_init_op=ready_for_local_init_op,
                     saver=model.saver)
  # After the session is created by the Supervisor and before the main while
  # loop:
  if is_chief and FLAGS.sync_replicas:
    sv.start_queue_runners(sess, [chief_queue_runner])
    # Insert initial tokens to the queue.
    sess.run(init_token_op)
  ```
  @@__init__
  @@compute_gradients
  @@apply_gradients
  @@get_chief_queue_runner
  @@get_init_tokens_op
  """

  def __init__(self,
               opt,
               global_step,
               total_num_replicas=None,
               variable_averages=None,
               variables_to_average=None,
               use_locking=False,
               name="sync_replicas"):
    """Construct a sync_replicas optimizer.
    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be one of the Optimizer classes.
      replicas_to_aggregate: number of replicas to aggregate for each variable
        update.
      total_num_replicas: Total number of tasks/workers/replicas, could be
        different from replicas_to_aggregate.
        If total_num_replicas > replicas_to_aggregate: it is backup_replicas +
        replicas_to_aggregate.
        If total_num_replicas < replicas_to_aggregate: Replicas compute
        multiple batches per update to variables.
      variable_averages: Optional `ExponentialMovingAverage` object, used to
        maintain moving averages for the variables passed in
        `variables_to_average`.
      variables_to_average: a list of variables that need to be averaged. Only
        needed if variable_averages is passed in.
      use_locking: If True use locks for update operation.
      name: string. Optional name of the returned operation.
    """
    if total_num_replicas is None:
      total_num_replicas = replicas_to_aggregate

    super(TimeoutReplicasOptimizer, self).__init__(use_locking, name)
    logging.info(
        "TimeoutReplicas: total_num_replicas=%s",
        total_num_replicas)
    self._n_updates = 0
    self._opt = opt
    self._gradients_applied = False
    self._variable_averages = variable_averages
    self._variables_to_average = variables_to_average
    self._total_num_replicas = total_num_replicas
    self._global_step = None

    # The synchronization op will be executed in a queue runner which should
    # only be executed by one of the replicas (usually the chief).
    self._chief_queue_runner = None

    # Remember which accumulator is on which device to set the initial step in
    # the accumulator to be global step. This list contains list of the
    # following format: (accumulator, device).
    self._accumulator_list = []

    # For timeout, we have one token queue per worker. This makes it so that
    # a worker can not take the work of another worker if it finishes early.
    self._sync_token_queues = [0] * self._total_num_replicas
    for worker in range(self._total_num_replicas):
      with ops.device(global_step.device):
        self._sync_token_queues[worker] = data_flow_ops.FIFOQueue(-1,
                                                                  global_step.dtype.base_dtype,
                                                                  shapes=(),
                                                                  shared_name="sync_token_q_%d" % worker)

  def start_interval_updates(self, sess, timeout_client):
    def interval_update():
      tf.logging.info("Interval update...")
      sess.run([self._update_op])
      timeout_client.broadcast_parameters_updated(self._n_updates)
      self._n_updates += 1
      Timer(FLAGS.interval_ms / float(1000), interval_update).start()
    Timer(FLAGS.interval_ms / float(1000), interval_update).start()

  def compute_gradients(self, *args, **kwargs):
    """Compute gradients of "loss" for the variables in "var_list".
    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping with per replica global norm if needed.
    The global norm with aggregated gradients can be bad as one replica's huge
    gradients can hurt the gradients from other replicas.
    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().
    Returns:
      A list of (gradient, variable) pairs.
    """
    return self._opt.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, worker_id, global_step=None, name=None, collect_cdfs=False):
    """Apply gradients to variables.
    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer.
    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.
    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and start the next one. This is executed by each replica.
    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")

    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    self._global_step = global_step
    train_ops = []
    aggregated_grad = []
    var_list = []

    self._local_step = variables.Variable(
        initial_value=0,
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES],
        dtype=global_step.dtype.base_dtype,
        name="sync_rep_local_step")
    self.local_step_init_op = state_ops.assign(self._local_step, global_step._ref())
    chief_init_ops = [self.local_step_init_op]
    self.ready_for_local_init_op = variables.report_uninitialized_variables(
      variables.all_variables())

    # The wait op waits for the current worker to dequeue a token from its respective token queue
    self._wait_op = self._sync_token_queues[worker_id].dequeue()

    # Replicas have to wait until they can get a token from the token queue
    # BEFORE begining to compute gradients.
    with ops.device(global_step.device):
      queue_size = self._sync_token_queues[worker_id].size()
      update_local_step_op = state_ops.assign(self._local_step, global_step._ref())

    # Gradient accum creation
    with ops.name_scope(None, self._name):
      for grad, var in grads_and_vars:
        var_list.append(var)
        tf.logging.info("Grad " + str(grad) + " assigned to " + str(var.device))
        with ops.device(var.device):
          if grad is None:
            continue
          elif isinstance(grad, ops.Tensor):
            grad_accum = data_flow_ops.ConditionalAccumulator(
              grad.dtype,
              shape=var.get_shape(),
              shared_name=var.name + "/grad_accum")
          else:
            if not isinstance(grad, ops.IndexedSlices):
              raise ValueError("Unknown grad type!")
            grad_accum = data_flow_ops.SparseConditionalAccumulator(
              grad.dtype, shape=(), shared_name=var.name + "/grad_accum")

          self._accumulator_list.append((grad_accum, var))

      # Phase 1 gradient computation
      with ops.control_dependencies([update_local_step_op]):
        for index, (grad, var) in enumerate(grads_and_vars):
          print_start_op = logging_ops.Print(global_step, [global_step], message="Starting to apply grads for variable %d" % index)
          with ops.device(var.device):
            if grad is None:
              continue

            elif isinstance(grad, ops.Tensor):
              grad_accum = self._accumulator_list[index][0]

              with ops.control_dependencies([print_start_op]):
                with tf.device("job:worker/task:0"):
                  apply_grad_op = grad_accum.apply_grad(grad,
                                                        local_step=self._local_step._ref())
                  with ops.control_dependencies([apply_grad_op]):
                    finished_print_op = logging_ops.Print(global_step, [global_step], message="Done applying grads for variable %d" % index)
                    train_ops.append(finished_print_op)

            else:
              if not isinstance(grad, ops.IndexedSlices):
                raise ValueError("Unknown grad type!")
              grad_accum = self._accumulator_list[index][0]


              with ops.control_dependencies([print_start_op]):
                with tf.device("job:worker/task:0"):
                  apply_grad_op = grad_accum.apply_indexed_slices_grad(
                    grad, local_step=self._local_step._ref())
                  with ops.control_dependencies([apply_grad_op]):
                    finished_print_op = logging_ops.Print(global_step, [global_step], message="Done applying grads for variable %d" % index)
                    train_ops.append(finished_print_op)

      # Phase 2 gradient applying
      for index, (grad, var) in enumerate(grads_and_vars):
        with ops.device(var.device):
          grad_accum = self._accumulator_list[index][0]
          if grad is None:
            aggregated_grad.append(None)
          elif isinstance(grad, ops.Tensor):
            if collect_cdfs:
              aggregated_grad.append(grad_accum.take_grad(self._total_num_replicas))
            else:
              aggregated_grad.append(grad_accum.take_grad(1))
          else:
            if collect_cdfs:
              aggregated_grad.append(grad_accum.take_grad(self._total_num_replicas))
            else:
              aggregated_grad.append(grad_accum.take_indexed_slices_grad(1))

      aggregated_grads_and_vars = zip(aggregated_grad, var_list)

      # Some debug operations
      self.print_sizes = logging_ops.Print(global_step, [self._sync_token_queues[i].size() for i in range(self._total_num_replicas)], message="queue sizes")
      self.print_accum_sizes = logging_ops.Print(self._local_step,
                                                 [x[0].num_accumulated() for x in self._accumulator_list] + [worker_id],
                                                 message="Accum sizes")
      self.print_local_step = logging_ops.Print(self._local_step, [self._local_step._ref(), global_step._ref()], message="local vs global step")

      # sync_op will be assigned to the same device as the global step.
      with ops.device(global_step.device), ops.name_scope(""):
        with ops.control_dependencies([self.print_accum_sizes]):
          update_op = self._opt.apply_gradients(aggregated_grads_and_vars, global_step)
          self._update_op = update_op
          with ops.control_dependencies([update_op]):
            sync_op = []
            for cur_worker_id in range(self._total_num_replicas):
              sync_op.append(self._sync_token_queues[cur_worker_id].enqueue(global_step))
            sync_op = control_flow_ops.group(*(sync_op))

        self.sync_op_ = sync_op

        # dummy_queue is passed to the queue runner. Don't use the real queues
        # because the queue runner doesn't automatically reopen it once it
        # closed queues in PS devices.
        dummy_queue = (
            data_flow_ops.FIFOQueue(1,
                                    types_pb2.DT_INT32,
                                    shapes=(),
                                    shared_name="dummy_queue"))

        self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                            [sync_op])

      with ops.device(global_step.device), ops.name_scope(""):
        with ops.control_dependencies(train_ops):
          # Worker finished applying gradients. Add token to phase1_finished_queue
          train_op = logging_ops.Print(self._local_step._ref(),
                                       [x[0].num_accumulated() for x in self._accumulator_list] + [worker_id],
                                       message="Finished worker updates",
                                       name="FinishedWorkerUpdatesPrint")


      for accum, var in self._accumulator_list:
        with ops.device(var.device):
          chief_init_ops.append(
              accum.set_global_step(
                  global_step, name="SetGlobalStep"))
      self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
      self._gradients_applied = True

      return train_op

  def get_chief_queue_runner(self):
    """Returns the QueueRunner for the chief to execute.
    This includes the operations to synchronize replicas: aggregate gradients,
    apply to variables, increment global step, insert tokens to token queue.
    Note that this can only be called after calling apply_gradients() which
    actually generates this queuerunner.
    Returns:
      A `QueueRunner` for chief to execute.
    Raises:
      ValueError: If this is called before apply_gradients().
    """
    if self._gradients_applied is False:
      raise ValueError("Should be called after apply_gradients().")

    return self._chief_queue_runner

  def get_slot(self, *args, **kwargs):
    """Return a slot named "name" created for "var" by the Optimizer.
    This simply wraps the get_slot() from the actual optimizer.
    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().
    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def get_slot_names(self, *args, **kwargs):
    """Return a list of the names of slots created by the `Optimizer`.
    This simply wraps the get_slot_names() from the actual optimizer.
    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().
    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def get_init_tokens_op(self, num_tokens=-1):
    """Returns the op to fill the sync_token_queue with the tokens.
    This is supposed to be executed in the beginning of the chief/sync thread
    so that even if the total_num_replicas is less than replicas_to_aggregate,
    the model can still proceed as the replicas can compute multiple steps per
    variable update. Make sure:
    `num_tokens >= replicas_to_aggregate - total_num_replicas`.
    Args:
      num_tokens: Number of tokens to add to the queue.
    Returns:
      An op for the chief/sync replica to fill the token queue.
    Raises:
      ValueError: If this is called before apply_gradients().
      ValueError: If num_tokens are smaller than replicas_to_aggregate -
        total_num_replicas.
    """
    if self._gradients_applied is False:
      raise ValueError(
          "get_init_tokens_op() should be called after apply_gradients().")

    tokens_needed = self._total_num_replicas
    if num_tokens == -1:
      num_tokens = self._total_num_replicas
    elif num_tokens < tokens_needed:
      raise ValueError(
          "Too few tokens to finish the first step: %d (given) vs %d (needed)" %
          (num_tokens, tokens_needed))

    init_tokens = []
    with ops.device(self._global_step.device), ops.name_scope(""):
      tokens = array_ops.fill([num_tokens],
                              self._global_step)
      for i in range(self._total_num_replicas):
        with ops.control_dependencies([logging_ops.Print(self._global_step, [self._global_step], message="Init token queue")]):
          init_tokens_op = self._sync_token_queues[i].enqueue(self._global_step)
        init_tokens.append(init_tokens_op)

    return init_tokens
