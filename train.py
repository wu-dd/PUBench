import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.data import Subset, ConcatDataset

from data import datasets
from core import hparams_registry, algorithms
from lib import misc

# python -m train --data_dir /home/kaefer/data/CIFAR10/ --dataset CIFAR10 --algorithm uPU --hparams_seed 0 --trial_seed 2 --seed 1331381724 --output_dir ./results/tmp/e85674511698ee7fcc3b4df11669d158 --holdout_fraction 0.1 --skip_model_save --setting set1_2 --calibration False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Positive-unlabeled learning')
    parser.add_argument('--data_dir', type=str, default="/home/kaefer/data/CIFAR10/")#
    parser.add_argument('--dataset', type=str, default="CIFAR10")#
    parser.add_argument('--algorithm', type=str, default="nPU")#
    parser.add_argument('--hparams', type=str,help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,help='Seed for random hparams (0 means "default hparams")')#
    parser.add_argument('--trial_seed', type=int, default=0,help='Trial number (used for seeding split_dataset and ''random_hparams).')#
    parser.add_argument('--seed', type=int, default=0,help='Seed for everything else')#
    parser.add_argument('--steps', type=int, default=20000,help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=100,help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.1)#
    parser.add_argument('--tabular_test_fraction', type=float, default=0.1, required=False)
    parser.add_argument('--skip_model_save', action='store_true') #
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--setting', help='Predefined setting: set1_1(2)-set5_1(2) use u_type one_sample; set6_1(2)-set10_1(2) use two_sample', default='set1_1', type=str, required=False)
    parser.add_argument('--calibration', help='whether to use calibration or not', type=lambda x: str(x).lower() in ['true', '1', 'yes'],default=False,) #
    args = parser.parse_args()
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    args = misc.gen_configs(args)
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    hparams['max_steps']=args.steps
    hparams['output_dir']=args.output_dir
    hparams['pi'] = args.pi
    hparams['dataset'] = args.dataset
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in datasets.IMAGE_DATASETS and args.dataset in vars(datasets):
        p_dataset, u_dataset, test_dataset = vars(datasets)[args.dataset](args)
    elif args.dataset in datasets.TABULAR_DATASETS and args.dataset in vars(datasets):
        p_dataset, u_dataset, test_dataset = vars(datasets)[args.dataset](args)
    else:
        raise NotImplementedError

    val_p_dataset, train_p_dataset = misc.split_dataset(p_dataset, int(len(p_dataset)*args.holdout_fraction), misc.seed_hash(args.trial_seed))
    val_u_dataset, train_u_dataset = misc.split_dataset(u_dataset, int(len(u_dataset)*args.holdout_fraction), misc.seed_hash(args.trial_seed))

    train_p_loader = torch.utils.data.DataLoader(dataset=train_p_dataset, batch_size=hparams['batch_size'], shuffle=True,num_workers=0, drop_last=True)
    val_p_loader = torch.utils.data.DataLoader(dataset=val_p_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    train_u_loader = torch.utils.data.DataLoader(dataset=train_u_dataset, batch_size=hparams['batch_size'], shuffle=True,num_workers=0, drop_last=True)
    val_u_loader = torch.utils.data.DataLoader(dataset=val_u_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

    if args.algorithm == 'CVIR':
        train_u_loader_eval = torch.utils.data.DataLoader(dataset=train_u_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=0, drop_last=False,pin_memory=True)
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(train_p_dataset.shape, train_u_dataset.shape, hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_p_minibatches_iterator = iter(train_p_loader)
    train_u_minibatches_iterator = iter(train_u_loader)

    checkpoint_vals = collections.defaultdict(lambda: [])
    steps_per_epoch_u = len(train_u_dataset) // hparams['batch_size']
    n_steps = args.steps
    ## WDD - top ##
    warmup_steps = hparams['warmup_steps'] if 'warmup_steps' in hparams else 0
    ## WDD - bottom ##
    checkpoint_freq = args.checkpoint_freq

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            # "model_input_shape": train_dataset.data.shape,
            # "model_num_classes": dataset.num_classes,
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    epoch = 0
    last_results_keys = None

    for step in range(start_step, warmup_steps+n_steps):
        epoch = step / steps_per_epoch_u
        algorithm.train()
        step_start_time = time.time()
        if args.algorithm == "Robust_PU" and algorithm.check_update_weight(epoch):
            algorithm.update_weight(train_p_dataset, train_u_dataset, epoch)
            # algorithm.update_weight_p(train_p_loader, epoch)
            # algorithm.update_weight_u(train_u_loader, epoch)
        #print('number of unlabeled data')
        #print(len(train_u_dataset))
        try:
            if args.algorithm == 'CVIR' and step % steps_per_epoch_u == 0:
                algorithm.before_train_epoch(train_u_loader_eval, device, len(train_u_dataset))
            p_idx, (inputs_p_w1, inputs_p_s), targets_p = next(train_p_minibatches_iterator)
            u_idx, (inputs_u_w1, inputs_u_s), (targets_u, _) = next(train_u_minibatches_iterator)

            # union calibration
            if args.calibration and args.u_type == 'one_sample':
                u_idx = torch.cat((u_idx, p_idx), dim=0)
                inputs_u_w1 = torch.cat((inputs_u_w1, inputs_p_w1), dim=0)
                inputs_u_s = torch.cat((inputs_u_s, inputs_p_s), dim=0)
                targets_p_u = targets_p - targets_p
                targets_u = torch.cat((targets_u, targets_p_u), dim=0)

            inputs_p_w1 = inputs_p_w1.to(device)
            inputs_p_s = inputs_p_s.to(device)
            targets_p = targets_p.to(device)
            inputs_u_w1 = inputs_u_w1.to(device)
            inputs_u_s = inputs_u_s.to(device)
            targets_u = targets_u.to(device)
            #targets_u_true = targets_u_true.to(device)

        except:
            if args.algorithm == 'CVIR' and step % steps_per_epoch_u == 0:
                # update pseudo label at the beginning of each epoch
                algorithm.before_train_epoch(train_u_loader_eval, device,len(train_u_dataset))
            train_p_minibatches_iterator = iter(train_p_loader)
            train_u_minibatches_iterator = iter(train_u_loader)

            p_idx, (inputs_p_w1, inputs_p_s), targets_p = next(train_p_minibatches_iterator)
            u_idx, (inputs_u_w1, inputs_u_s), (targets_u, targets_u_true) = next(train_u_minibatches_iterator)

            # union calibration
            if args.calibration and args.u_type == 'one_sample':
                u_idx = torch.cat((u_idx, p_idx), dim=0)
                inputs_u_w1 = torch.cat((inputs_u_w1, inputs_p_w1), dim=0)
                inputs_u_s = torch.cat((inputs_u_s, inputs_p_s), dim=0)
                targets_p_u = targets_p - targets_p
                targets_u = torch.cat((targets_u, targets_p_u), dim=0)

            inputs_p_w1 = inputs_p_w1.to(device)
            inputs_p_s = inputs_p_s.to(device)
            targets_p = targets_p.to(device)
            inputs_u_w1 = inputs_u_w1.to(device)
            inputs_u_s = inputs_u_s.to(device)
            targets_u = targets_u.to(device)
            #targets_u_true = targets_u_true.to(device)

        ## WDD - top ##
        if args.algorithm == 'CVIR':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u,u_idx)
        elif args.algorithm == 'Dist_PU':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, step+1, inputs_p_s, inputs_u_s, step // steps_per_epoch_u, n_steps // steps_per_epoch_u)
        elif args.algorithm == 'PUSB':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, step, train_p_loader, train_u_loader)
        elif args.algorithm == 'PUe':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, step, train_p_loader)
        elif args.algorithm == 'LBE':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, step, test_loader)
        elif args.algorithm == 'HolisticPU':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, step, inputs_p_s, inputs_u_s, u_idx, train_u_loader)
        elif args.algorithm == 'PULDA':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, step, p_idx, u_idx, train_p_loader, train_u_loader)
        elif args.algorithm == 'PAN':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, epoch)
        elif args.algorithm == 'P3MIX_C':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, epoch, train_p_dataset)
        elif args.algorithm == 'P3MIX_E':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, epoch, train_p_dataset)
        elif args.algorithm == 'Robust_PU':
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u, epoch, p_idx, u_idx)
        else:
            step_vals = algorithm.update(inputs_p_w1, targets_p, inputs_u_w1, targets_u)
        
        """
        if args.algorithm == 'CVIR':
            step_vals = algorithm.update(inputs_p_w1, inputs_p_s, targets_p, inputs_u_w1, inputs_u_s, targets_u,u_idx)
        elif args.algorithm == 'Dist_PU':
            #print(step // steps_per_epoch_u, n_steps // steps_per_epoch_u,steps_per_epoch_u)
            step_vals = algorithm.update(inputs_p_w1, inputs_p_s, targets_p, inputs_u_w1, inputs_u_s, targets_u,step // steps_per_epoch_u, n_steps // steps_per_epoch_u)
            if (step + 1) % steps_per_epoch_u ==0:
                algorithm.after_train_epoch((step + 1) / steps_per_epoch_u)
        elif args.algorithm == 'GLWS':
            # input two weak augmentations for GLWS
            step_vals = algorithm.update(inputs_p_w1, inputs_p_w2, targets_p, inputs_u_w1, inputs_u_w2, targets_u)

        else:
            step_vals = algorithm.update(inputs_p_w1, inputs_p_s, targets_p, inputs_u_w1, inputs_u_s, targets_u)
        """
        ## WDD - bottom ##

        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # validation
            val_oracle_acc = misc.val_oracle_accuracy(algorithm, val_p_loader, val_u_loader, device, args.u_type)
            results['val_o_acc'] = val_oracle_acc
            val_proxy_acc = misc.val_proxy_accuracy(algorithm, val_p_loader, val_u_loader, device, args.pi, args.u_type)
            results['val_p_acc'] = val_proxy_acc
            val_proxy_auc = misc.val_proxy_auc(algorithm, val_p_loader, val_u_loader, device)
            results['val_p_auc'] = val_proxy_auc

            #acc = misc.accuracy(algorithm, test_loader, device)
            #results['test_acc'] = acc
            test_results = misc.compute_metrics(algorithm, test_loader, device)
            results['test_acc'] = test_results['Accuracy']
            results['test_precision'] = test_results['Precision']
            results['test_recall'] = test_results['Recall']
            results['test_f1'] = test_results['F1']
            results['test_auc'] = test_results['AUC']

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, cls=misc.NpEncoder, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            #start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')