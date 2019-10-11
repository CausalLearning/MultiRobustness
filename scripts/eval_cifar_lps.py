import tensorflow as tf
import numpy as np
from pgd_attack import PGDAttack, compute_grad
from cifar10_model import Model
from scripts.utils import get_ckpt
from eval_ch import evaluate_ch, get_model, get_saver
from eval_fb import evaluate_fb
from eval import evaluate
from multiprocessing import Pool
import sys


models_slim = [
]

models_wide = [
    ('path_to_model', -1),
]

attack_configs = [
        {"type": "linf", "epsilon": 4.0, "k": 100, "random_start": True, "reps": 20},
        {"type": "linf", "epsilon": 4.0, "k": 1000, "random_start": True},
        {"type": "l1", "epsilon": 2000, "k": 100, "random_start": True, "perc": 99, "a": 2.0, "reps": 20},
        {"type": "l1", "epsilon": 2000, "k": 1000, "random_start": True, "perc": 99, "a": 2.0}
]

outdir = "cifar_" + str(int(attack_configs[0]["epsilon"]))

eval_config = {"data": "cifar10",
               "data_path": "cifar10_data",
               "num_eval_examples": 1000,
               "eval_batch_size": 100}

eval_wide = sys.argv[1] == "wide"

if eval_wide:
    models = models_wide
    eval_config["filters"] = [16, 160, 320, 640]
else:
    models = models_slim
    eval_config["filters"] = [16, 16, 32, 64]

model = Model(eval_config)
grad = compute_grad(model)
attacks = [PGDAttack(model, a_config, 0.0, 255.0, grad) for a_config in attack_configs]

saver = tf.train.Saver()
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.gpu_options.per_process_gpu_memory_fraction = 1.0

nat_accs = np.zeros(len(models))
adv_accs = np.zeros((len(models), len(attacks) + 2))

any_attack = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)
any_l1 = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)
any_linf = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)

def worker((model_dir, epoch)):

    model_name = model_dir.split('/')[-1] + "_" + str(epoch)
    output_file = "results/{}/lps/{}_l1_fb.npy".format(outdir, model_name)

    try:
        all_corr_adv1 = np.load(output_file)
        return all_corr_adv1
    except:

        g = tf.Graph()
        with g.as_default():
            config_tf = tf.ConfigProto()
            config_tf.gpu_options.allow_growth = True
            config_tf.gpu_options.per_process_gpu_memory_fraction = 0.2
            with tf.Session(graph=g, config=config_tf) as sess:
                model = Model(eval_config)

                saver = tf.train.Saver()
                ckpt = get_ckpt(model_dir, epoch)
                print("loading ", ckpt)
                saver.restore(sess, ckpt)

                # FB attacks
                print("Foolbox l1 attack")
                all_corr_nat1, all_corr_adv1, l1s = evaluate_fb(model, eval_config, 0.0, 255.0, norm='l1', bound=2000, verbose=False)
                all_corr_adv1 = all_corr_adv1 | ((l1s > 2000) & all_corr_nat1)
        
                np.save(output_file, all_corr_adv1)

                return all_corr_adv1

pool = Pool(max(len(models), 4))
all_models_corr_adv1 = pool.map(worker, models)
pool.close()
pool.join()

adv_accs[:, len(attacks)] = np.mean(all_models_corr_adv1, axis=-1)
any_attack &= all_models_corr_adv1
any_l1 &= all_models_corr_adv1

print("DONE with FB!")

print(nat_accs)
print(adv_accs)
print("any: ", np.mean(any_attack, axis=-1))
print("l1: ", np.mean(any_l1, axis=-1))
print("linf: ", np.mean(any_linf, axis=-1))

with tf.Session(config=config_tf) as sess:
    for m_idx, (model_dir, epoch) in enumerate(models):
        ckpt = get_ckpt(model_dir, epoch)
        saver.restore(sess, ckpt)

        print("starting...", model_dir, epoch)

        # lp attacks
        nat_acc, total_corr_advs = evaluate(model, attacks, sess, eval_config)
        nat_accs[m_idx] = nat_acc
        adv_acc = np.mean(total_corr_advs, axis=-1)
        adv_accs[m_idx, :len(attacks)] = adv_acc
        any_attack[m_idx] &= np.bitwise_and.reduce(np.asarray(total_corr_advs), 0)

        print(model_dir, adv_accs[m_idx])
        model_name = models[m_idx][0].split('/')[-1] + "_" + str(models[m_idx][1])
        for i, attack in enumerate(attacks):
            np.save("results/{}/lps/{}_{}.npy".format(outdir, model_name, attack.name), total_corr_advs[i])

            if attack_configs[i]["type"] == "l1":
                any_l1[m_idx] &= total_corr_advs[i]
            else:
                any_linf[m_idx] &= total_corr_advs[i]

print("DONE with PGD!")
print(nat_accs)
print(adv_accs)
print("any: ", np.mean(any_attack, axis=-1))
print("l1: ", np.mean(any_l1, axis=-1))
print("linf: ", np.mean(any_linf, axis=-1))

tf.reset_default_graph()

# Cleverhans attacks
g2 = tf.Graph()
with g2.as_default():
    with tf.Session(graph=g2, config=config_tf) as sess2:
        model2 = get_model(eval_config)
        saver2 = get_saver(eval_config)

        for m_idx, (model_dir, epoch) in enumerate(models):
            ckpt = get_ckpt(model_dir, epoch)
            saver.restore(sess2, ckpt)

            print("starting...", model_dir, epoch)

            print("EAD")
            all_corr_nat1, all_corr_adv1, l1s = evaluate_ch(model2, eval_config, sess2, "l1", 2000, verbose=True)
            all_corr_adv1 = all_corr_adv1 | ((l1s > 2000) & all_corr_nat1)
            adv_accs[m_idx, len(attacks) + 1] = np.mean(all_corr_adv1)
            any_attack[m_idx] &= all_corr_adv1

            print(model_dir, adv_accs[m_idx])

            model_name = models[m_idx][0].split('/')[-1] + "_" + str(models[m_idx][1])
            any_l1[m_idx] &= all_corr_adv1
            np.save("results/{}/lps/{}_l1_ead.npy".format(outdir, model_name), all_corr_adv1)

print(nat_accs)
print(adv_accs)
print("any: ", np.mean(any_attack, axis=-1))
print("l1: ", np.mean(any_l1, axis=-1))
print("linf: ", np.mean(any_linf, axis=-1))

