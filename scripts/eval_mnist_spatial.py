import tensorflow as tf
import numpy as np
from pgd_attack import PGDAttack, compute_grad
from model import Model
from scripts.utils import get_ckpt
from eval import evaluate


models = [
    ('path_to_model', -1),
]

attack_configs = [
        {"type": "linf", "epsilon": 0.3, "k": 100, "random_start": True, "reps": 40},
        {"type": "linf", "epsilon": 0.3, "k": 1000, "random_start": True},
        {"type": "RT", "spatial_limits": [3, 3, 30], "grid_granularity": [5, 5, 31], "random_tries": 10},
        {"type": "RT", "spatial_limits": [3, 3, 30], "grid_granularity": [5, 5, 31], "random_tries": -1}
]

model = Model({"model_type": "cnn"})
grad = compute_grad(model)
attacks = [PGDAttack(model, a_config, 0.0, 1.0, grad) for a_config in attack_configs]

saver = tf.train.Saver()
config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.gpu_options.per_process_gpu_memory_fraction = 0.2

eval_config = {"data": "mnist",
               "model_type": "cnn",
               "num_eval_examples": 200,
               "eval_batch_size": 100}

nat_accs = np.zeros(len(models))
adv_accs = np.zeros((len(models), len(attacks)))

any_attack = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)
any_rt = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)
any_linf = np.ones((len(models), eval_config["num_eval_examples"])).astype(np.bool)

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
        model_name = model_dir.split('/')[-1] + "_" + str(epoch)
        for i, attack in enumerate(attacks):
            np.save("results/mnist/spatial/{}_{}.npy".format(model_name, attack.name), total_corr_advs[i])

            if attack_configs[i]["type"] == "RT":
                any_rt[m_idx] &= total_corr_advs[i]
            else:
                any_linf[m_idx] &= total_corr_advs[i]

print(nat_accs)
print(adv_accs)
print("any: ", np.mean(any_attack, axis=-1))
print("rt: ", np.mean(any_rt, axis=-1))
print("linf: ", np.mean(any_linf, axis=-1))
