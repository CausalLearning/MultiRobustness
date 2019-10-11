from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os

import numpy as np
import tensorflow as tf
import argparse

from cleverhans.utils import set_log_level
from cleverhans.attacks import ElasticNetMethod, CarliniWagnerL2
from cleverhans.evaluation import batch_eval
import logging


def one_hot(a, n_classes):
    res = np.zeros((len(a), n_classes), dtype=np.int64)
    res[np.arange(len(a)), a] = 1
    return res


def evaluate_ch(model, config, sess, norm='l1', bound=None, verbose=True):
    dataset = config['data']
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']

    if dataset == "mnist":
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
        X = mnist.test.images[0:num_eval_examples, :].reshape(-1, 28, 28, 1)
        Y = mnist.test.labels[0:num_eval_examples]
        x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    else:
        import cifar10_input
        data_path = config["data_path"]
        cifar = cifar10_input.CIFAR10Data(data_path)
        X = cifar.eval_data.xs[0:num_eval_examples, :].astype(np.float32) / 255.0
        Y = cifar.eval_data.ys[0:num_eval_examples]
        x_image = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        assert norm == 'l1'

    if norm=='l2':
        attack = CarliniWagnerL2(model, sess)
        params = {'batch_size': eval_batch_size, 'binary_search_steps': 9}
    else:
        attack = ElasticNetMethod(model, sess, clip_min=0.0, clip_max=1.0)
        params = {'beta': 1e-2,
                  'decision_rule': 'L1',
                  'batch_size': eval_batch_size,
                  'learning_rate': 1e-2,
                  'max_iterations': 1000}

    if verbose:
        set_log_level(logging.DEBUG, name="cleverhans")
    
    y = tf.placeholder(tf.int64, shape=[None, 10])
    params['y'] = y
    adv_x = attack.generate(x_image, **params)
    preds_adv = model.get_predicted_class(adv_x)
    preds_nat = model.get_predicted_class(x_image)

    all_preds, all_preds_adv, all_adv_x = batch_eval(
        sess, [x_image, y], [preds_nat, preds_adv, adv_x], [X, one_hot(Y, 10)], batch_size=eval_batch_size)

    print('acc nat', np.mean(all_preds == Y))
    print('acc adv', np.mean(all_preds_adv == Y))

    if dataset == "cifar10":
        X *= 255.0
        all_adv_x *= 255.0

    if norm == 'l2':
        lps = np.sqrt(np.sum(np.square(all_adv_x - X), axis=(1,2,3)))
    else:
        lps = np.sum(np.abs(all_adv_x - X), axis=(1,2,3))
    print('mean lp: ', np.mean(lps))
    for b in [bound, bound/2.0, bound/4.0, bound/8.0]:
        print('lp={}, acc={}'.format(b, np.mean((all_preds_adv == Y) | (lps > b))))

    all_corr_adv = (all_preds_adv == Y)
    all_corr_nat = (all_preds == Y)
    return all_corr_nat, all_corr_adv, lps


def get_model(config):
    dataset = config["data"]
    if dataset == "mnist":
        from cleverhans_models import MadryMNIST
        model = MadryMNIST()
    else:
        from cleverhans_models import make_wresnet
        model = make_wresnet(scope="a", filters=config["filters"])

    return model 


def get_saver(config):
    dataset = config["data"]
    if dataset == "cifar10":
        # nasty hack
        gvars = tf.global_variables()
        saver = tf.train.Saver({v.name[2:-2]: v for v in gvars if v.name[:2] == "a/"})
    else:
        saver = tf.train.Saver()
    return saver


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Eval script options',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model_dir', type=str,
                        help='path to model directory')
    parser.add_argument('--epoch', type=int, default=None,
                        help='specific epoch to load (default=latest)')
    parser.add_argument('--eval_cpu', help='evaluate on CPU',
                        action="store_true")
    parser.add_argument('--norm', help='norm to use', choices=['l1', 'l2'], default='l1')
    parser.add_argument('--bound', type=float, help='attack noise bound', default=None)

    args = parser.parse_args()

    if args.eval_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    model_dir = args.model_dir

    with open(model_dir + "/config.json") as config_file:
        config = json.load(config_file)

    model = get_model(config)
    saver = get_saver(config)

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
    config_tf.gpu_options.per_process_gpu_memory_fraction = 0.1

    with tf.Session(config=config_tf) as sess:
        # Restore the checkpoint
        print('Evaluating checkpoint {}'.format(cur_checkpoint))
        saver.restore(sess, cur_checkpoint)

        evaluate_ch(model, config, sess, args.norm, args.bound)
