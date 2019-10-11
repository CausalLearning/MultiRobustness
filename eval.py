"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import math
import os

import numpy as np
import tensorflow as tf
import argparse

from pgd_attack import PGDAttack, compute_grad

rows = cols = 8


def show_images(images, cols=1, figpath="figure.png"):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    n_images = len(images)
    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        if np.max(image) > 1.0:
            image = image.astype(np.uint8)

        plt.imshow(image)
    plt.savefig(figpath)
    plt.close()


# A function for evaluating a single checkpoint
def evaluate(model, eval_attacks, sess, config, plot=False, summary_writer=None, eval_train=False, eval_validation=False, verbose=True):
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']

    dataset = config["data"]
    assert dataset in ["mnist", "cifar10"]

    if dataset == "mnist":
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        if "model_type" in config and config["model_type"] == "linear":
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
    else:
        import cifar10_input
        data_path = config["data_path"]
        cifar = cifar10_input.CIFAR10Data(data_path)

    np.random.seed(0)
    tf.random.set_random_seed(0)
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_advs = np.zeros(len(eval_attacks), dtype=np.float32)
    total_corr_nat = 0.
    total_corr_advs = [[] for _ in range(len(eval_attacks))]

    l1_norms = [[] for _ in range(len(eval_attacks))]
    l2_norms = [[] for _ in range(len(eval_attacks))]
    linf_norms = [[] for _ in range(len(eval_attacks))]

    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        if eval_train:
            if dataset == "mnist":
                x_batch = mnist.train.images[bstart:bend, :].reshape(-1, 28, 28, 1)
                y_batch = mnist.train.labels[bstart:bend]
            else:
                x_batch = cifar.train_data.xs[bstart:bend, :].astype(np.float32)
                y_batch = cifar.train_data.ys[bstart:bend]
        elif eval_validation:
            assert dataset == "cifar10"
            offset = len(cifar.eval_data.ys) - num_eval_examples
            x_batch = cifar.eval_data.xs[offset+bstart:offset+bend, :].astype(np.float32)
            y_batch = cifar.eval_data.ys[offset+bstart:offset+bend]

        else:
            if dataset == "mnist":
                x_batch = mnist.test.images[bstart:bend, :].reshape(-1, 28, 28, 1)
                y_batch = mnist.test.labels[bstart:bend]
            else:
                x_batch = cifar.eval_data.xs[bstart:bend, :].astype(np.float32)
                y_batch = cifar.eval_data.ys[bstart:bend]

        noop_trans = np.zeros([len(x_batch), 3])
        dict_nat = {model.x_input: x_batch,
                    model.y_input: y_batch,
                    model.is_training: False,
                    model.transform: noop_trans}

        cur_corr_nat, cur_xent_nat = sess.run(
            [model.num_correct, model.xent],
            feed_dict=dict_nat)

        total_xent_nat += cur_xent_nat
        total_corr_nat += cur_corr_nat

        for i, attack in enumerate(eval_attacks):
            x_batch_adv, adv_trans = attack.perturb(x_batch, y_batch, sess)

            dict_adv = {model.x_input: x_batch_adv,
                        model.y_input: y_batch,
                        model.is_training: False,
                        model.transform: adv_trans if adv_trans is not None else np.zeros([len(x_batch), 3])}

            cur_corr_adv, cur_xent_adv, cur_corr_pred, cur_adv_images = \
                sess.run([model.num_correct, model.xent, model.correct_prediction, model.x_image],
                         feed_dict=dict_adv)

            total_xent_advs[i] += cur_xent_adv
            total_corr_advs[i].extend(cur_corr_pred)

            l1_norms[i].extend(np.sum(np.abs(x_batch_adv - x_batch).reshape(len(x_batch), -1), axis=-1))
            l2_norms[i].extend(np.linalg.norm((x_batch_adv - x_batch).reshape(len(x_batch), -1), axis=-1))
            linf_norms[i].extend(np.max(np.abs(x_batch_adv - x_batch).reshape(len(x_batch), -1), axis=-1))

    avg_xent_nat = total_xent_nat / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples

    avg_xent_advs = total_xent_advs / num_eval_examples
    acc_advs = np.sum(total_corr_advs, axis=-1) / num_eval_examples

    if len(eval_attacks) > 0:
        tot_correct = np.bitwise_and.reduce(np.asarray(total_corr_advs), 0)
        assert len(tot_correct) == num_eval_examples
        any_acc = np.sum(tot_correct) / num_eval_examples

    if verbose:
        print('natural: {:.2f}%'.format(100 * acc_nat))
        for i, attack in enumerate(eval_attacks):
            t = attack.name
            print('adversarial ({}):'.format(t))
            print('\tacc: {:.2f}% '.format(100 * acc_advs[i]))
            print("\tmean(l1)={:.1f}, min(l1)={:.1f}, max(l1)={:.1f}".format(
                np.mean(l1_norms[i]), np.min(l1_norms[i]), np.max(l1_norms[i])))
            print("\tmean(l2)={:.1f}, min(l2)={:.1f}, max(l2)={:.1f}".format(
                np.mean(l2_norms[i]), np.min(l2_norms[i]), np.max(l2_norms[i])))
            print("\tmean(linf)={:.1f}, min(linf)={:.1f}, max(linf)={:.1f}".format(
                np.mean(linf_norms[i]), np.min(linf_norms[i]), np.max(linf_norms[i])))

        print('avg nat loss: {:.2f}'.format(avg_xent_nat))
        for i, attack in enumerate(eval_attacks):
            t = attack.name
            print('avg adv loss ({}): {:.2f}'.format(t, avg_xent_advs[i]))

        if len(eval_attacks) > 0:
            print("any attack: {:.2f}%".format(100 * any_acc))

    if summary_writer:

        values = [
            tf.Summary.Value(tag='xent nat', simple_value=avg_xent_nat),
            tf.Summary.Value(tag='accuracy nat', simple_value=acc_nat)
        ]
        if len(eval_attacks) > 0:
            values.append(tf.Summary.Value(tag='accuracy adv any', simple_value=any_acc))

        for i, attack in enumerate(eval_attacks):
            t = attack.name
            adv_values = [
                tf.Summary.Value(tag='xent adv eval ({})'.format(t), simple_value=avg_xent_advs[i]),
                tf.Summary.Value(tag='xent adv ({})'.format(t), simple_value=avg_xent_advs[i]),
                tf.Summary.Value(tag='accuracy adv eval ({})'.format(t), simple_value=acc_advs[i]),
                tf.Summary.Value(tag='accuracy adv ({})'.format(t), simple_value=acc_advs[i])
            ]
            values.extend(adv_values)

        summary = tf.Summary(value=values)
        summary_writer.add_summary(summary, global_step.eval(sess))

    return acc_nat, total_corr_advs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Eval script options',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_dir', type=str,
                        help='path to model directory')
    parser.add_argument('--epoch', type=int, default=None,
                        help='specific epoch to load (default=latest)')
    parser.add_argument('--eval_train', help='evaluate on training set',
                        action="store_true")
    parser.add_argument('--eval_cpu', help='evaluate on CPU',
                        action="store_true")
    args = parser.parse_args()

    if args.eval_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    model_dir = args.model_dir

    with open(model_dir + "/config.json") as config_file:
        config = json.load(config_file)

    eval_attack_configs = [np.asarray(config["attacks"])[i] for i in config["eval_attacks"]]
    print(eval_attack_configs)

    dataset = config["data"]
    if dataset == "mnist":
        from model import Model
        model = Model(config)

        x_min, x_max = 0.0, 1.0
    else:
        from cifar10_model import Model
        model = Model(config)
        x_min, x_max = 0.0, 255.0

    grad = compute_grad(model)
    eval_attacks = [PGDAttack(model, a_config, x_min, x_max, grad) for a_config in eval_attack_configs]

    global_step = tf.contrib.framework.get_or_create_global_step()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    eval_dir = os.path.join(model_dir, 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    saver = tf.train.Saver()

    if args.epoch is not None:
        ckpts = tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths
        ckpt = [c for c in ckpts if c.endswith('checkpoint-{}'.format(args.epoch))]
        assert len(ckpt) == 1
        cur_checkpoint = ckpt[0]
    else:
        cur_checkpoint = tf.train.latest_checkpoint(model_dir)
    assert cur_checkpoint is not None

    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    config_tf.gpu_options.per_process_gpu_memory_fraction = 1.0

    with tf.Session(config=config_tf) as sess:
        # Restore the checkpoint
        print('Evaluating checkpoint {}'.format(cur_checkpoint))

        saver.restore(sess, cur_checkpoint)

        evaluate(model, eval_attacks, sess, config, plot=True, eval_train=args.eval_train)

