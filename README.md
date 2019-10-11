# Adversarial Training and Robustness for Multiple Perturbations

Code for the paper:

**Adversarial Training and Robustness for Multiple Perturbations**<br>
*Florian Tramèr and Dan Boneh*<br>
Conference on Neural Information Processing Systems (NeurIPS), 2019<br>
https://arxiv.org/abs/1904.13000

Our work studies the scalability and effectiveness of adversarial training for achieving robustness against a combination of multiple types of adversarial examples.
We currently implement multiple Lp-bounded attacks (L1, L2, Linf) as well as rotation-translation attacks, for both MNIST and CIFAR10.

Before training a model, edit the `config.json` file to specify the training, attack, and evaluation parameters. The given `config.json` file can be used as a basis for MNIST experiments, while the `config_cifar10.json` file has the apropriate hyperparameters for CIFAR10.

## Training 

To train, simply run:

```[bash]
python train.py output/dir/
```
This will read the `config.json` file from the current directory, and save the trained model, logs, as well as the original config file into `output/dir/`.

## Evaluation

We performed a fairly thorough evaluation of the models we trained using a wide range of attacks. Unfortunately, there is currently no single library implementing all these attacks so we combined different ones. Some attacks we implemented ourselves (different forms of PGD and rotation-translation), others are taken from [Cleverhans](https://github.com/tensorflow/cleverhans) and from [Foolbox](https://github.com/bethgelab/foolbox).
Our [evaluation scripts](scripts/) can give you an idea of how we evaluate a model against all attacks.

## Config options

Many hyperparameters in the `config.json` file are standard and self-explanatory.
Specific to our work are the following parameters you may consider tuning:

* `"multi_attack_mode"`: When training against multiple attacks, this flag indicates whether to train against examples from all attacks (default), or only on the worst example for each input (`"MAX"`). For the wide ResNet model on CIFAR10, the default option causes memory overflow due to too large batches. The `"HALF_BATCH_HALF_LR"` flag halves the batch size (and the learning rate accordingly) to avoid overflows.

* `"attacks"`: This list specifies the attacks used for either training or evaluation (or both). The parameters are standard, except for our new L1 attack. This comes with a `"perc"` parameter that specifies the sparsity of the gradient updates (see the paper for detail), and a step-size multiplier (`"a"`). The value of the `"perc"` parameter can be a range (e.g., `[80, 99]`) in which case the sparsity of each gradient update in an attack is sampled uniformly from that range. Each attack can take a `"reps"` parameter (default: 1) that specifies the number of times an attack should be repeated. 

* `"train_attacks"` and `"eval_attacks"`: Specify which of the attacks defined under `"attacks"` should be used for training or evaluation. These are lists of indices into `"attacks"`. I.e., `"train_attacks": [0, 1, 2]` means that the first 3 defined attacks are used for training.</br>
Our paper also defines a new type of *affine attack* that interpolates between two attack types. You can specify an affine attack via a tuple of attacks: e.g., `"eval_attacks": [0, [1, 2]]` will evaluate against the first attack, and against an affine attack that interpolates between the second and third attack. The weighting used by the affine attack can be specified by adding a `"weight"` parameter to the attack parameters.

## Acknowledgments
Parts of the codebase are inspired or directly borrowed from:
* https://github.com/MadryLab/cifar10_challenge
* https://github.com/MadryLab/adversarial_spatial


## Citation

If our code or our results are useful in your reasearch, please consider citing:

```[bibtex]
@inproceedings{TB19,
  author={Tram{\`e}r, Florian and Boneh, Dan},
  title={Adversarial Training and Robustness for Multiple Perturbations},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2019},
  howpublished={arXiv preprint arXiv:1904.13000},
  url={https://arxiv.org/abs/1904.13000}
}
```

