"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from itertools import product
from collections import Counter
import json


def uniform_weights(n_attacks, n_samples):
    x = np.random.uniform(size=(n_attacks, n_samples))
    y = np.maximum(-np.log(x), 1e-8)
    return y / np.sum(y, axis=0, keepdims=True)


def init_delta(x, attack, weight):
    if not attack["random_start"]:
        return np.zeros_like(x)

    assert len(weight) == len(x)
    eps = (attack["epsilon"] * weight).reshape(len(x), 1, 1, 1)

    if attack["type"] == "linf":
        return np.random.uniform(-eps, eps, x.shape)
    elif attack["type"] == "l2":
        r = np.random.randn(*x.shape)
        norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1).reshape(-1, 1, 1, 1)
        return (r / norm) * eps
    elif attack["type"] == "l1":
        r = np.random.laplace(size=x.shape)
        norm = np.linalg.norm(r.reshape(r.shape[0], -1), axis=-1, ord=1).reshape(-1, 1, 1, 1)
        return (r / norm) * eps
    else:
        raise ValueError("Unknown norm {}".format(attack["type"]))


def delta_update(old_delta, g, x_adv, attack, x_min, x_max, weight, seed=None, t=None):
    assert len(weight) == len(x_adv)

    eps_w = attack["epsilon"] * weight
    eps = eps_w.reshape(len(x_adv), 1, 1, 1)

    if attack["type"] == "linf":
        a = attack.get('a', (2.5 * eps) / attack["k"])
        new_delta = old_delta + a * np.sign(g)
        new_delta = np.clip(new_delta, -eps, eps)

        new_delta = np.clip(new_delta, x_min - (x_adv - old_delta), x_max - (x_adv - old_delta))
        return new_delta

    elif attack["type"] == "l2":
        a = attack.get('a', (2.5 * eps) / attack["k"])
        bad_pos = ((x_adv == x_max) & (g > 0)) | ((x_adv == x_min) & (g < 0))
        g[bad_pos] = 0

        g = g.reshape(len(g), -1)
        g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
        g = g.reshape(old_delta.shape)

        new_delta = old_delta + a * g
        new_delta_norm = np.linalg.norm(new_delta.reshape(len(new_delta), -1), axis=-1).reshape(-1, 1, 1, 1)
        new_delta = new_delta / np.maximum(new_delta_norm, 1e-8) * np.minimum(new_delta_norm, eps)
        new_delta = np.clip(new_delta, x_min - (x_adv - old_delta), x_max - (x_adv - old_delta))
        return new_delta

    elif attack["type"] == "l1":
        _, h, w, ch = g.shape

        a = attack.get('a', 1.0) * x_max
        perc = attack.get('perc', 99)

        if perc == 'max':
            bad_pos = ((x_adv > (x_max - a)) & (g > 0)) | ((x_adv < a) & (g < 0)) | (x_adv < x_min) | (x_adv > x_max)
            g[bad_pos] = 0
        else:
            bad_pos = ((x_adv == x_max) & (g > 0)) | ((x_adv == x_min) & (g < 0))
            g[bad_pos] = 0

        abs_grad = np.abs(g)
        sign = np.sign(g)

        if perc == 'max':
            grad_flat = abs_grad.reshape(len(abs_grad), -1)
            max_abs_grad = np.argmax(grad_flat, axis=-1)
            optimal_perturbation = np.zeros_like(grad_flat)
            optimal_perturbation[np.arange(len(grad_flat)), max_abs_grad] = 1.0
            optimal_perturbation = sign * optimal_perturbation.reshape(abs_grad.shape)
        else:
            if isinstance(perc, list):
                perc_low, perc_high = perc
                perc = np.random.RandomState(seed).uniform(low=perc_low, high=perc_high)

            max_abs_grad = np.percentile(abs_grad, perc, axis=(1, 2, 3), keepdims=True)
            tied_for_max = (abs_grad >= max_abs_grad).astype(np.float32)
            num_ties = np.sum(tied_for_max, (1, 2, 3), keepdims=True)
            optimal_perturbation = sign * tied_for_max / num_ties

        new_delta = old_delta + a * optimal_perturbation

        l1 = np.sum(np.abs(new_delta), axis=(1, 2, 3))
        to_project = l1 > eps_w
        if np.any(to_project):
            n = np.sum(to_project)
            d = new_delta[to_project].reshape(n, -1)  # n * N (N=h*w*ch)
            abs_d = np.abs(d)  # n * N
            mu = -np.sort(-abs_d, axis=-1)  # n * N
            cumsums = mu.cumsum(axis=-1)  # n * N
            eps_d = eps_w[to_project]
            js = 1.0 / np.arange(1, h * w * ch + 1)
            temp = mu - js * (cumsums - np.expand_dims(eps_d, -1))
            rho = np.argmin(temp > 0, axis=-1)
            theta = 1.0 / (1 + rho) * (cumsums[range(n), rho] - eps_d)
            sgn = np.sign(d)
            d = sgn * np.maximum(abs_d - np.expand_dims(theta, -1), 0)
            new_delta[to_project] = d.reshape(-1, h, w, ch)

        new_delta = np.clip(new_delta, x_min - (x_adv - old_delta), x_max - (x_adv - old_delta))
        return new_delta


def compute_grad(model):
    label_mask = tf.one_hot(model.y_input,
                            model.pre_softmax.get_shape().as_list()[-1],
                            on_value=1.0,
                            off_value=0.0,
                            dtype=tf.float32)
    correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
    wrong_logit = tf.reduce_max((1 - label_mask) * model.pre_softmax - 1e4 * label_mask, axis=1)
    loss = -(correct_logit - wrong_logit)
    return tf.gradients(loss, model.x_input)[0]


def name(attack):
    return json.dumps(attack)


class PGDAttack:
    def __init__(self, model, attack_config, x_min, x_max, grad, reps=1):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        print("new attack: ", attack_config)
        if isinstance(attack_config, dict):
            attack_config = [attack_config]

        self.model = model
        self.x_min = x_min
        self.x_max = x_max
        self.attack_config = attack_config
        self.names = [name(a) for a in attack_config]
        self.name = " - ".join(self.names)
        self.grad = grad
        self.reps = int(attack_config[0].get("reps", 1))
        assert self.reps >= 1

    def perturb(self, x_nat, y, sess, x_nat_no_aug=None):

        if len(self.attack_config) == 0:
            return x_nat, None

        if x_nat_no_aug is None:
            x_nat_no_aug = x_nat

        n = len(x_nat)
        worst_x = np.copy(x_nat)
        worst_t = np.zeros([n, 3])
        max_xent = np.zeros(n)
        all_correct = np.ones(n).astype(bool)

        for i in range(self.reps):
            if "weight" in self.attack_config[0]:
                weights = np.asarray([a["weight"] for a in self.attack_config])
                weights = np.repeat(weights[:, np.newaxis], len(x_nat), axis=-1)
            else:
                weights = uniform_weights(len(self.attack_config), len(x_nat))

            if self.attack_config[0]["type"] == "RT":
                assert np.all([a["type"] != "RT" for a in self.attack_config[1:]])
                norm_attacks = self.attack_config[1:]
                norm_weights = weights[1:]
                x_adv, trans = self.grid_perturb(x_nat_no_aug, y, sess, self.attack_config[0], 
                                                 weights[0], norm_attacks, norm_weights)
            else:
                # rotation and translation attack should always come first
                assert np.all([a["type"] != "RT" for a in self.attack_config])
                norm_attacks = self.attack_config
                x_adv = self.norm_perturb(x_nat, y, sess, norm_attacks, weights)
                trans = worst_t

            cur_xent, cur_correct = sess.run([self.model.y_xent, self.model.correct_prediction],
                                             feed_dict={self.model.x_input: x_adv,
                                                        self.model.y_input: y,
                                                        self.model.is_training: False,
                                                        self.model.transform: trans})
            cur_xent = np.asarray(cur_xent)
            cur_correct = np.asarray(cur_correct)

            idx = (cur_xent > max_xent) & (cur_correct == all_correct)
            idx = idx | (cur_correct < all_correct)
            max_xent = np.maximum(cur_xent, max_xent)
            all_correct = cur_correct & all_correct

            idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1)
            worst_t = np.where(idx, trans, worst_t)  # shape (bsize, 3)

            idx = np.expand_dims(idx, axis=-1)
            idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
            worst_x = np.where(idx, x_adv, worst_x, )  # shape (bsize, h, w, ch)

        return worst_x, worst_t

    def grid_perturb(self, x_nat, y, sess, attack_config, weight, norm_attacks, norm_weights):
        random_tries = attack_config["random_tries"]
        n = len(x_nat)

        assert len(weight) == len(x_nat)
        # (3, 1) * n => (3, n)
        spatial_limits = np.asarray(attack_config["spatial_limits"])[:, np.newaxis] * weight

        if random_tries > 0:
            grids = np.zeros((n, random_tries))
        else:
            # exhaustive grid
            # n * (num_x * num_y * num_rot)
            grids = [list(product(*list(np.linspace(-l, l, num=g)
                                        for l, g in zip(spatial_limits[:, i], attack_config["grid_granularity"]))))
                     for i in range(len(x_nat))]
            grids = np.asarray(grids)

        worst_x = np.copy(x_nat)
        worst_t = np.zeros([n, 3])
        max_xent = np.zeros(n)
        all_correct = np.ones(n).astype(bool)

        for idx in range(len(grids[0])):
            if random_tries > 0:
                t = [[np.random.uniform(-l, l) for l in spatial_limits[:, i]] for i in range(len(x_nat))]
            else:
                t = grids[:, idx]

            x = self.norm_perturb(x_nat, y, sess, norm_attacks, norm_weights, trans=t)

            curr_dict = {self.model.x_input: x,
                         self.model.y_input: y,
                         self.model.is_training: False,
                         self.model.transform: t}

            cur_xent, cur_correct = sess.run([self.model.y_xent,
                                              self.model.correct_prediction],
                                             feed_dict=curr_dict)  # shape (bsize,)
            cur_xent = np.asarray(cur_xent)
            cur_correct = np.asarray(cur_correct)

            # Select indices to update: we choose the misclassified transformation
            # of maximum xent (or just highest xent if everything else if correct).
            idx = (cur_xent > max_xent) & (cur_correct == all_correct)
            idx = idx | (cur_correct < all_correct)
            max_xent = np.maximum(cur_xent, max_xent)
            all_correct = cur_correct & all_correct

            idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1)
            worst_t = np.where(idx, t, worst_t)  # shape (bsize, 3)

            idx = np.expand_dims(idx, axis=-1)
            idx = np.expand_dims(idx, axis=-1)  # shape (bsize, 1, 1, 1)
            worst_x = np.where(idx, x, worst_x, )  # shape (bsize, h, w, ch)

        return worst_x, worst_t

    def norm_perturb(self, x_nat, y, sess, norm_attacks, norm_weights, trans=None):
        if len(norm_attacks) == 0:
            return x_nat

        x_min = self.x_min
        x_max = self.x_max

        if trans is None:
            trans = np.zeros([len(x_nat), 3])

        iters = [a["k"] for a in norm_attacks]
        assert (np.all(np.asarray(iters) == iters[0]))

        deltas = np.asarray([init_delta(x_nat, attack, weight)
                             for attack, weight in zip(norm_attacks, norm_weights)])
        x_adv = np.clip(x_nat + np.sum(deltas, axis=0), 0, 1)


        # a seed that remains constant across attack iterations
        seed = np.random.randint(low=0, high=2**32-1)

        for i in range(np.sum(iters)):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x_adv,
                                                  self.model.y_input: y,
                                                  self.model.is_training: False,
                                                  self.model.transform: trans})

            deltas[i % len(norm_attacks)] = delta_update(deltas[i % len(norm_attacks)],
                                                         grad,
                                                         x_adv,
                                                         norm_attacks[i % len(norm_attacks)],
                                                         x_min, x_max,
                                                         norm_weights[i % len(norm_attacks)],
                                                         seed=seed, t=i+1)

            x_adv = np.clip(x_nat + np.sum(deltas, axis=0), x_min, x_max)

        return np.clip(x_nat + np.sum(deltas, axis=0), x_min, x_max)
