import tensorflow as tf
import numpy as np
from pgd_attack import PGDAttack, compute_grad
from model import Model
from scripts.utils import get_ckpt
from eval import evaluate
from eval_fb import evaluate_fb
from eval_ch import evaluate_ch, get_model, get_saver
from eval_bapp import evaluate_bapp
from multiprocessing import Pool
import os


models = [
    ('path_to_model', -1),
]

attack_configs = [
    {"type": "linf", "epsilon": 0.3, "k": 100, "random_start": True, "reps": 40},
    {"type": "l1", "epsilon": 10, "k": 100, "random_start": True, "perc": 99, "a": 0.5, "reps": 40},
    {"type": "l2", "epsilon": 2, "k": 100, "random_start": True, "reps": 40},
]

model = Model({"model_type": "cnn"})
grad = compute_grad(model)
attacks = [PGDAttack(model, a_config, 0.0, 1.0, grad) for a_config in attack_configs]

saver = tf.train.Saver()
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.gpu_options.per_process_gpu_memory_fraction = 0.5


eval_config = {"data": "mnist", 
               "num_eval_examples": 200,
               "eval_batch_size": 200}

nat_accs = np.zeros(len(models))
adv_accs = np.zeros((len(models), len(attacks) + 5))

any_attack = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)
any_l1 = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)
any_l2 = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)
any_linf = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)

def worker((model_dir, epoch)):
    g = tf.Graph()
    with g.as_default():
        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        config_tf.gpu_options.per_process_gpu_memory_fraction = 0.2
        with tf.Session(graph=g, config=config_tf) as sess:
            model = Model({"model_type": "cnn"})

            saver = tf.train.Saver()
            ckpt = get_ckpt(model_dir, epoch)
            saver.restore(sess, ckpt)

            # FB attacks
            print("Foolbox l1 attack")
            all_corr_nat1, all_corr_adv1, l1s = evaluate_fb(model, eval_config, 0.0, 1.0, norm='l1', bound=10, verbose=False)
            all_corr_adv1 = all_corr_adv1 | ((l1s > 10) & all_corr_nat1)

            print("Foolbox l2 attack")
            all_corr_nat2, all_corr_adv2, l2s = evaluate_fb(model, eval_config, 0.0, 1.0, norm='l2', bound=2.0, verbose=False)
            all_corr_adv2 = all_corr_adv2 | ((l2s > 2.0) & all_corr_nat2)
            return all_corr_adv1, all_corr_adv2


pool = Pool(4)
all_models_corr_adv = pool.map(worker, models)

all_models_corr_adv1 = np.asarray([a[0] for a in all_models_corr_adv])
all_models_corr_adv2 = np.asarray([a[1] for a in all_models_corr_adv])

adv_accs[:, len(attacks)] = np.mean(all_models_corr_adv1, axis=-1)
any_attack &= all_models_corr_adv1
any_l1 &= all_models_corr_adv1
adv_accs[:, len(attacks) + 1] = np.mean(all_models_corr_adv2, axis=-1)
any_attack &= all_models_corr_adv2
any_l2 &= all_models_corr_adv2

print("DONE with FB!")

print(nat_accs)
print(adv_accs)
print("any: ", np.mean(any_attack, axis=-1))
print("l1: ", np.mean(any_l1, axis=-1))
print("l2: ", np.mean(any_l2, axis=-1))
print("linf: ", np.mean(any_linf, axis=-1))

with tf.Session(config=config_tf) as sess:
    for m_idx, (model_dir, epoch) in enumerate(models):
        ckpt = get_ckpt(model_dir, epoch)
        saver.restore(sess, ckpt)

        print("starting...", model_dir)

        # lp attacks
        nat_acc, total_corr_advs = evaluate(model, attacks, sess, eval_config)
        nat_accs[m_idx] = nat_acc
        adv_acc = np.mean(total_corr_advs, axis=-1)
        adv_accs[m_idx, :len(attacks)] = adv_acc
        any_attack[m_idx] &= np.bitwise_and.reduce(np.asarray(total_corr_advs), 0)

        print(model_dir, adv_accs[m_idx])
        model_name = model_dir.split('/')[-1]
        for i, attack in enumerate(attacks):
            np.save("results/mnist/{}_{}.npy".format(model_name, attack.name), total_corr_advs[i])

            if attack_configs[i]["type"] == "l1":
                any_l1[m_idx] &= total_corr_advs[i]
            elif attack_configs[i]["type"] == "l2":
                any_l2[m_idx] &= total_corr_advs[i]
            else:
                any_linf[m_idx] &= total_corr_advs[i]

print("DONE with PGD!")
print(nat_accs)
print(adv_accs)
print("any: ", np.mean(any_attack, axis=-1))
print("l1: ", np.mean(any_l1, axis=-1))
print("l2: ", np.mean(any_l2, axis=-1))
print("linf: ", np.mean(any_linf, axis=-1))

with tf.Session(config=config_tf) as sess:
    for m_idx, (model_dir, epoch) in enumerate(models):
        ckpt = get_ckpt(model_dir, epoch)
        saver.restore(sess, ckpt)

        print("starting...", model_dir)
        all_corr_nat_inf, all_corr_adv_inf, l_infs = evaluate_bapp(sess, model, eval_config, 0, 1, 0.3, verbose=False)
        all_corr_adv_inf = all_corr_adv_inf | ((l_infs > 0.3) & all_corr_nat_inf)
        adv_accs[m_idx, len(attacks) + 2] = np.mean(all_corr_adv_inf)
        any_attack[m_idx] &= all_corr_adv_inf

        any_linf[m_idx] &= all_corr_adv_inf

# Cleverhans attacks
g2 = tf.Graph()
with g2.as_default():
    with tf.Session(graph=g2, config=config_tf) as sess2:
        model2 = get_model(eval_config)
        saver2 = get_saver(eval_config)

        for m_idx, (model_dir, epoch) in enumerate(models):
            ckpt = get_ckpt(model_dir, epoch)
            saver.restore(sess2, ckpt)

            print("starting...", model_dir)

            print("EAD")
            all_corr_nat1, all_corr_adv1, l1s = evaluate_ch(model2, eval_config, sess2, "l1", 10, verbose=False)
            all_corr_adv1 = all_corr_adv1 | ((l1s > 10) & all_corr_nat1)
            adv_accs[m_idx, len(attacks) + 3] = np.mean(all_corr_adv1)
            any_attack[m_idx] &= all_corr_adv1

            print("C&W")
            all_corr_nat2, all_corr_adv2, l2s = evaluate_ch(model2, eval_config, sess2, "l2", 2, verbose=False)
            all_corr_adv2 = all_corr_adv2 | ((l2s > 2.0) & all_corr_nat2)
            adv_accs[m_idx, len(attacks) + 4] = np.mean(all_corr_adv2)
            any_attack[m_idx] &= all_corr_adv2

            print(model_dir, adv_accs[m_idx])

            model_name = model_dir.split('/')[-1]
            any_l1[m_idx] &= all_corr_adv1 
            any_l2[m_idx] &= all_corr_adv2 
            np.save("results/mnist/{}_l1_ead.npy".format(model_name), all_corr_adv1)
            np.save("results/mnist/{}_l2_cw.npy".format(model_name), all_corr_adv2)

print(nat_accs)
print(adv_accs)
print("any: ", np.mean(any_attack, axis=-1))
print("l1: ", np.mean(any_l1, axis=-1))
print("l2: ", np.mean(any_l2, axis=-1))
print("linf: ", np.mean(any_linf, axis=-1))
