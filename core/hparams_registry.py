import numpy as np
from lib import misc

MLP_DATASET = [
    "Letter",
    "USPS",
    "Creditcard"
]

RESNET_DATASET = [
    "CIFAR10",
    "IMAGENETTE"
]


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    if dataset in MLP_DATASET:
        _hparam('model', 'MLP', lambda r: 'MLP')
    elif dataset in RESNET_DATASET:
        _hparam('model', 'ResNet', lambda r: 'ResNet')

    _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
    _hparam('momentum', 0.9, lambda r: 0.9)
    if dataset in RESNET_DATASET:
        _hparam('batch_size', 64, lambda r: 2**int(r.uniform(5, 8)))
    elif dataset in MLP_DATASET:
        if dataset == "Creditcard":
            _hparam('batch_size', 32, lambda r: 2**int(r.uniform(4, 6)))
        else:
            _hparam('batch_size', 64, lambda r: 2**int(r.uniform(4, 7)))

    # algorithm-specific hyperparameters
    if algorithm == 'nnPU':
        _hparam('beta', 0., lambda r: 0.)
    ## WDD - top ##
    elif algorithm == 'PUe':
        _hparam('warmup_steps',2000, lambda r:2000)
        _hparam('alpha',0.1, lambda r:0.1)
    elif algorithm == 'PUbN':
        _hparam('tau',0.5, lambda r: r.choice([0.5,0.7,0.9]))
    elif algorithm == 'LBE':
        _hparam('warmup_steps',2000, lambda r:2000)
    elif algorithm =='HolisticPU':
        _hparam('warmup_steps',2000, lambda r:2000)
        _hparam('rho', 0.1, lambda r: 0.1)
        _hparam('use_ema',True,lambda r: [True])
        _hparam('mu',1, lambda r: 1)
        _hparam('T',1, lambda r:1) # pseudo label temperature
    elif algorithm == 'PULDA':
        _hparam('use_ema',True,lambda r: [True])
        _hparam('tmpr',3.5,lambda r: 3.5)
        _hparam('alpha', 11, lambda r: 11)
        _hparam('alpha_U',0.85,lambda r: 0.85)
        _hparam('alpha_CN',0.5,lambda r: 0.5)
        _hparam('two_way', 1, lambda r: r.choice([0, 1]))
        _hparam('margin', 0.6, lambda r: 0.6)
        _hparam('co_mixup',4.2, lambda r: 4.2)
        _hparam('warmup_steps',2000, lambda r:2000)
    ## WDD - bottom ##
    elif algorithm == 'PAN':
        _hparam('l', 0.0001, lambda r: 0.0001)
    elif algorithm == 'P3MIX_C':
        _hparam('start_hmix', 5, lambda r: 5)
        _hparam('gamma', 0.8, lambda r: 0.8)
        _hparam('alpha', 1.0, lambda r: 1.0)
        _hparam('top_k', 96, lambda r: 96)
        _hparam('positive_weight', 1., lambda r: 1.)
        _hparam('unlabeled_weight', 1., lambda r: 1.)
        _hparam('entropy_weight', 0.1, lambda r: 0.1)
        _hparam('milestones', [20, 40, 60, 80, 100], lambda r: [20, 40, 60, 80, 100])
        _hparam('scheduler_gamma', 0.5, lambda r: 0.5)
    elif algorithm == 'P3MIX_E':
        _hparam('start_hmix', 5, lambda r: 5)
        _hparam('gamma', 0.85, lambda r: 0.85)
        _hparam('top_k', 96, lambda r: 96)
        _hparam('ema_start', 0, lambda r: 0)
        _hparam('positive_weight', 1., lambda r: 1.)
        _hparam('unlabeled_weight', 1., lambda r: 1.)
        _hparam('entropy_weight', 0.5, lambda r: 0.5)
        _hparam('elr_weight', 5., lambda r: 5.)
        _hparam('ema_decay', 0.997, lambda r: 0.997)
        _hparam('milestones', [20, 40, 60, 80, 100], lambda r: [20, 40, 60, 80, 100])
        _hparam('scheduler_gamma', 0.5, lambda r: 0.5)
    elif algorithm == 'Robust_PU':
        _hparam('temper_n', 1.3, lambda r: 1.)
        _hparam('temper_p', 1., lambda r: 1.)
        _hparam('scheduler_type_n', 'linear', lambda r: r.choice(['linear']))
        _hparam('scheduler_type_p', 'linear', lambda r: r.choice(['linear']))
        _hparam('alpha_n', 0.11, lambda r: 0.1)
        _hparam('alpha_p', 0.1, lambda r: 0.1)
        _hparam('max_thresh_n', 1., lambda r: 2.)
        _hparam('max_thresh_p', 1., lambda r: 2.)
        _hparam('grow_steps_n', 5, lambda r: 10)
        _hparam('grow_steps_p', 5, lambda r: 10)
        _hparam('pretrain_epoch', 10, lambda r: 10)
        _hparam('freq', 5, lambda r: 5)
        # _hparam('patience', 3, lambda r: 3)
        _hparam('lr_pretrain', 0.0001, lambda r: 0.0001)
    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
