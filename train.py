"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from pgd_attack import PGDAttack, compute_grad
from eval import evaluate

import sys
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

model_dir = sys.argv[1]

try:
    with open(model_dir + "/config.json") as config_file:
        config = json.load(config_file)
        print("opened previous config file")
except IOError:
    with open("config.json") as config_file:
        config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']

batch_size = config['training_batch_size']

dataset = config["data"]
assert dataset in ["mnist", "cifar10"]

num_train_attacks = len(config["train_attacks"])
multi_attack_mode = config["multi_attack_mode"]
print("num_train_attacks", num_train_attacks)
print("multi_attack_mode", multi_attack_mode)

step_size_schedule = config['step_size_schedule']
step_size_schedule = np.asarray(step_size_schedule)

# strategies for training with adversarial examples from K attacks:
#
# HALF_LR:      Keeps the clean batch size fixed 
#               (so the effective batch size is multiplied by K) and divides the learning rate by K
#
# HALF_BATCH:   Divides the clean batch size by K (so the ffective batch size remains unchanged). 
#               This is necessary to avoid memory overflows with the wide ResNet model on CIFAR10
#
if "HALF_LR" in multi_attack_mode:
    step_size_schedule[:, 1] *= 1. / num_train_attacks
if "HALF_BATCH" in multi_attack_mode or "ALTERNATE" in multi_attack_mode: 
    step_size_schedule[:, 0] *= num_train_attacks
    max_num_training_steps *= num_train_attacks
    max_num_training_steps = int(max_num_training_steps)

if "HALF_BATCH" in multi_attack_mode:
    batch_size *= 1. / num_train_attacks
    batch_size = int(batch_size)
print("batch_size", batch_size)

boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
global_step = tf.contrib.framework.get_or_create_global_step()
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32), boundaries, values)

if dataset == "mnist":
    from tensorflow.examples.tutorials.mnist import input_data
    from model import Model

    # Setting up the data and the model
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    num_train_data = 60000
    if config["model_type"] == "linear":
        x_train = mnist.train.images
        y_train = mnist.train.labels
        x_test = mnist.test.images
        y_test = mnist.test.labels

        pos_train = (y_train == 5) | (y_train == 7)
        x_train = x_train[pos_train]
        y_train = y_train[pos_train]
        y_train = (y_train == 5).astype(np.int64)
        pos_test = (y_test == 5) | (y_test == 7)
        x_test = x_test[pos_test]
        y_test = y_test[pos_test]
        y_test = (y_test == 5).astype(np.int64)

        from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
        from tensorflow.contrib.learn.python.learn.datasets import base 
        
        options = dict(dtype=tf.uint8, reshape=False, seed=None)
        train = DataSet(x_train, y_train, **options)
        test = DataSet(x_test, y_test, **options)

        mnist = base.Datasets(train=train, validation=None, test=test)
        num_train_data = len(x_train)

    model = Model(config)
    x_min, x_max = 0.0, 1.0

    # Setting up the optimizer
    opt = tf.train.AdamOptimizer(learning_rate)
    gv = opt.compute_gradients(model.xent)
    train_step = opt.apply_gradients(gv, global_step=global_step)
else:
    import cifar10_input
    from cifar10_model import Model

    weight_decay = config['weight_decay']
    data_path = config['data_path']
    momentum = config['momentum']
    raw_cifar = cifar10_input.CIFAR10Data(data_path)
    num_train_data = 50000
    model = Model(config)
    x_min, x_max = 0.0, 255.0

    # Setting up the optimizer
    total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
    opt = tf.train.MomentumOptimizer(learning_rate, momentum)
    gv = opt.compute_gradients(total_loss)
    train_step = opt.apply_gradients(gv, global_step=global_step)

num_epochs = (max_num_training_steps * batch_size) // num_train_data
print("num_epochs: {:d}".format(num_epochs))
print("max_num_training_steps", max_num_training_steps)
print("step_size_schedule", step_size_schedule)

# Set up adversary
grad = compute_grad(model)
train_attack_configs = [np.asarray(config["attacks"])[i] for i in config["train_attacks"]]
eval_attack_configs = [np.asarray(config["attacks"])[i] for i in config["eval_attacks"]]
train_attacks = [PGDAttack(model, a_config, x_min, x_max, grad) for a_config in train_attack_configs]

# Optimization that works well on MNIST: do a first epoch with a lower epsilon
start_small = config.get("start_small", False)
if start_small:
    train_attack_configs_small = [a.copy() for a in train_attack_configs]
    for attack in train_attack_configs_small:
        if 'epsilon' in attack:
            attack['epsilon'] /= 3.0
        else:
            attack['spatial_limits'] = [s/3.0 for s in attack['spatial_limits']]
    train_attacks_small = [PGDAttack(model, a_config, x_min, x_max, grad) for a_config in train_attack_configs_small] 
print('start_small', start_small)

eval_attacks = [PGDAttack(model, a_config, x_min, x_max, grad) for a_config in eval_attack_configs]

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    shutil.copy('config.json', model_dir)

eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

train_dir = os.path.join(model_dir, 'train')
if not os.path.exists(train_dir):
    os.makedirs(train_dir)

saver = tf.train.Saver(max_to_keep=100)
tf.summary.scalar('accuracy adv train', model.accuracy, collections=['adv'])
tf.summary.scalar('xent adv train', model.mean_xent, collections=['adv'])
tf.summary.image('images adv train', model.x_image, collections=['adv'])
adv_summaries = tf.summary.merge_all('adv')

tf.summary.scalar('accuracy_nat_train', model.accuracy, collections=['nat'])
tf.summary.scalar('xent_nat_train', model.mean_xent, collections=['nat'])
tf.summary.scalar('learning_rate', learning_rate, collections=['nat'])
nat_summaries = tf.summary.merge_all('nat')

eval_summaries_train = []
for i, attack in enumerate(eval_attacks):
    a_type = attack.name
    tf.summary.scalar('accuracy adv train {}'.format(a_type), model.accuracy, collections=['adv_{}'.format(i)])
    tf.summary.scalar('xent adv train {}'.format(a_type), model.mean_xent, collections=['adv_{}'.format(i)])
    tf.summary.image('images adv train {}'.format(a_type), model.x_image, collections=['adv_{}'.format(i)])
    eval_summaries_train.append(tf.summary.merge_all('adv_{}'.format(i)))

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
if dataset == "mnist":
    config_tf.gpu_options.per_process_gpu_memory_fraction = 0.2
else:
    config_tf.gpu_options.per_process_gpu_memory_fraction = 1.0
    config_tf.allow_soft_placement = True

with tf.Session(config=config_tf) as sess:
    if dataset == "cifar10":
        # initialize data augmentation
        cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess)

    # Initialize the summary writer, global variables, and our time counter.
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
    test_summary_writer = tf.summary.FileWriter(eval_dir)
    sess.run(tf.global_variables_initializer())
    training_time = 0.0

    cur_checkpoint = tf.train.latest_checkpoint(model_dir)
    if cur_checkpoint is not None:
        saver.restore(sess, cur_checkpoint)
    else:
        print("no checkpoint to load")

    start_step = sess.run(global_step)

    # Main training loop
    for ii in range(start_step, max_num_training_steps + 1):
        curr_epoch = (ii * batch_size) // num_train_data

        if dataset == "mnist":
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            x_batch = x_batch.reshape(-1, 28, 28, 1)
            x_batch_no_aug = x_batch
        else:
            x_batch_no_aug, x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)
            x_batch_no_aug = x_batch_no_aug.astype(np.float32)
            x_batch = x_batch.astype(np.float32)

        noop_trans = np.zeros([len(x_batch), 3])

        if start_small and curr_epoch == 0:
            curr_train_attacks = train_attacks_small
        else:
            curr_train_attacks = train_attacks

        # Compute Adversarial Perturbations
        start = timer()
        if multi_attack_mode == "ALTERNATE":
            # alternate between attacks each batch (does not work verywell)
            curr_attack = curr_train_attacks[ii % num_train_attacks]
            adv_outputs = [curr_attack.perturb(x_batch, y_batch, sess, x_nat_no_aug=x_batch_no_aug)]

        elif multi_attack_mode == "MAX":
            # choose best attack for each input
            adv_outputs = [attack.perturb(x_batch, y_batch, sess, x_nat_no_aug=x_batch_no_aug) for attack in curr_train_attacks]
            losses = np.zeros((num_train_attacks, len(x_batch)))
            for j in range(num_train_attacks):
                x = adv_outputs[j][0]
                t = adv_outputs[j][1]
                losses[j] = sess.run(model.y_xent,
                                     feed_dict={model.x_input: x,
                                                model.y_input: y_batch,
                                                model.is_training: False,
                                                model.transform: t if t is not None else noop_trans})
            best_idx = np.argmax(losses, axis=0)  # shape (batch_size,)
            best_x = np.asarray([adv_outputs[best_idx[j]][0][j] for j in range(len(x_batch))]) 
            best_t = np.asarray([adv_outputs[best_idx[j]][1][j] for j in range(len(x_batch))])
            adv_outputs = [(best_x, best_t)]

        else:
            # concatenate multiple attacks (default)
            adv_outputs = [attack.perturb(x_batch, y_batch, sess, x_nat_no_aug=x_batch_no_aug) for attack in curr_train_attacks]

        x_batch_advs = [a[0] for a in adv_outputs]
        all_trans = [a[1] if a[1] is not None else noop_trans for a in adv_outputs]
        end = timer()
        training_time += end - start

        nat_dict = {model.x_input: x_batch,
                    model.y_input: y_batch,
                    model.is_training: False,
                    model.transform: noop_trans}

        if num_train_attacks > 0:
            x_batch_adv = np.concatenate(x_batch_advs)
            y_batch_adv = np.concatenate([y_batch for _ in range(len(x_batch_advs))])
            trans_adv = np.concatenate(all_trans)

            adv_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch_adv,
                        model.is_training: False,
                        model.transform: trans_adv}
        else:
            adv_dict = nat_dict

        if ii % num_output_steps == 0:
            print('Step {} (epoch {}):    ({})'.format(ii, curr_epoch, datetime.now()))
            if ii > 0:
                print('    {} examples per second'.format(num_output_steps * batch_size / training_time))
            training_time = 0.0
            summary = sess.run(adv_summaries, feed_dict=adv_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))
            summary = sess.run(nat_summaries, feed_dict=nat_dict)
            summary_writer.add_summary(summary, global_step.eval(sess))

        # Output to stdout and tensorboard summaries
        if ii % num_summary_steps == 0:
            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            print('    training nat accuracy {:.4}%'.format(nat_acc * 100))

            for a_idx, attack in enumerate(eval_attacks):
                x_batch_adv_eval, trans_eval = attack.perturb(x_batch, y_batch, sess, x_nat_no_aug=x_batch_no_aug)

                adv_dict_eval = {model.x_input: x_batch_adv_eval,
                                 model.y_input: y_batch,
                                 model.is_training: False,
                                 model.transform: trans_eval if trans_eval is not None else noop_trans}

                adv_acc = sess.run(model.accuracy, feed_dict=adv_dict_eval)
                print('    training adv accuracy ({}) {:.4}%'.format(attack.name, adv_acc * 100))

                summary = sess.run(eval_summaries_train[a_idx], feed_dict=adv_dict_eval)
                summary_writer.add_summary(summary, global_step.eval(sess))

            evaluate(model, eval_attacks, sess, config, plot=False,
                     summary_writer=test_summary_writer, eval_train=False)

        # Write a checkpoint
        if ii % num_checkpoint_steps == 0 and ii > 0:
            saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)

        # Actual training step
        start = timer()
        adv_dict[model.is_training] = True
        _, curr_gv = sess.run([train_step, [g for (g, v) in gv]], feed_dict=adv_dict)
        end = timer()
        training_time += end - start
