from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import json
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import argparse
import foolbox



def evaluate_fb(model, config, x_min, x_max, norm='l1', bound=None, verbose=True):
    fmodel = foolbox.models.TensorFlowModel(model.x_input, model.pre_softmax, (x_min, x_max))

    if norm == 'l2':
        attack = foolbox.attacks.BoundaryAttack(fmodel)
    else:
        attack = foolbox.attacks.PointwiseAttack(fmodel)

    dataset = config["data"]
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']

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

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    all_corr_nat = []
    all_corr_adv = []
    lps = []

    num_inconsistencies = 0
    num_solved_inconsistencies = 0

    pbar = tqdm(total=num_eval_examples)

    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        if dataset == "mnist":
            x_batch = mnist.test.images[bstart:bend, :].reshape(-1, 28, 28, 1)
            y_batch = mnist.test.labels[bstart:bend]
        else:
            x_batch = cifar.eval_data.xs[bstart:bend, :].astype(np.float32)
            y_batch = cifar.eval_data.ys[bstart:bend]

        adversarials = []
        preds_adv = []
        for x, y in zip(x_batch, y_batch):

            for trial in range(1):
                if norm == "l2":
                    adversarial = attack(x, y, iterations=5000, max_directions=25)
                else:
                    adversarial = attack(x, y)
                failed = False
                if adversarial is None:
                    failed = True
                    adversarial = x

                pred_adv = y
                if not failed:
                    pred_adv = np.argmax(fmodel.predictions(adversarial))
                    if pred_adv == y:
                        num_inconsistencies += 1
                        if verbose:
                            print("Inconsistency with l2 {:.3f}!".format(np.sqrt(np.sum(np.square(adversarial - x)))))
                        new_adversarials = np.asarray([x + a * (adversarial - x) for a in [1.001, 1.005, 1.01, 1.05, 1.1]])
                        new_preds_adv = np.argmax(fmodel.batch_predictions(new_adversarials), axis=-1)

                        if ((new_preds_adv == y)).all():
                            failed = True
                            adversarial = x
                            if verbose:
                                print("Failed to resolve inconsistency!")
                        else:
                            adversarial = new_adversarials[np.argmin(new_preds_adv != y)]
                            pred_adv = new_preds_adv[np.argmin(new_preds_adv != y)]
                            num_solved_inconsistencies += 1
                            if verbose:
                                print("Solved inconsistency")

                if norm == 'l1':
                    lp = np.sum(np.abs(adversarial - x))
                else:
                    lp = np.sqrt(np.sum(np.square(adversarial - x)))

                if verbose:
                    print("trial {}".format(trial), lp, failed)

                if lp < bound:
                    break
            lps.append(lp)
            adversarials.append(adversarial)
            preds_adv.append(pred_adv)
            if not verbose:
                pbar.update(n=1)

        preds = np.argmax(fmodel.batch_predictions(x_batch), axis=-1)
        all_corr_nat.extend(preds == y_batch)
        all_corr_adv.extend(preds_adv == y_batch)

        if verbose:
            all_corr_adv = np.asarray(all_corr_adv)
            all_corr_nat = np.asarray(all_corr_nat)
            lps = np.asarray(lps)
            print('acc adv w. bound', np.mean(all_corr_adv | ((lps > bound) & all_corr_nat)))

    pbar.close()

    all_corr_adv = np.asarray(all_corr_adv)
    all_corr_nat = np.asarray(all_corr_nat)
    lps = np.asarray(lps)

    acc_nat = np.mean(all_corr_nat)
    acc_adv = np.mean(all_corr_adv)
    print('acc_nat', acc_nat)
    print('acc_adv', acc_adv)
    print('min(lp)={:.2f}, max(lp)={:.2f}, mean(lp)={:.2f}, median(lp)={:.2f}'.format(
        np.min(lps), np.max(lps), np.mean(lps), np.median(lps)))
    print('acc adv w. bound', np.mean(all_corr_adv | ((lps > bound) & all_corr_nat)))

    print("num_inconsistencies", num_inconsistencies)
    print("num_solved_inconsistencies", num_solved_inconsistencies)

    return all_corr_nat, all_corr_adv, lps


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
    parser.add_argument('--bound', type=float, help='Foolbox pointwise attack noise bound', default=None)

    args = parser.parse_args()

    if args.eval_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    model_dir = args.model_dir

    with open(model_dir + "/config.json") as config_file:
        config = json.load(config_file)

    dataset = config["data"]
    if dataset == "mnist":
        from model import Model
        model = Model(config)

        x_min, x_max = 0.0, 1.0
    else:
        from cifar10_model import Model
        model = Model(config)
        x_min, x_max = 0.0, 255.0

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
    if dataset == "mnist":
        config_tf.gpu_options.per_process_gpu_memory_fraction = 0.1
    else:
        config_tf.gpu_options.per_process_gpu_memory_fraction = 0.1

    with tf.Session(config=config_tf) as sess:
        # Restore the checkpoint
        print('Evaluating checkpoint {}'.format(cur_checkpoint))
        saver.restore(sess, cur_checkpoint)

        evaluate_fb(model, config, x_min, x_max, args.norm, args.bound)

