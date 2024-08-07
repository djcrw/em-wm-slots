#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import data_utils as du
import model_utils as mu
import cell_analyses as ca
import parameters

import numpy as np
import torch
import torch.optim as optim

import glob
import os
import shutil
import model as _model_
import time
import importlib
import wandb

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

importlib.reload(_model_)

# Initialise hyper-parameters for model
params = parameters.default_params()

# Create directories for storing all information about the current run
run_path, train_path, model_path, save_path, script_path, envs_path, run_name = du.make_directories(
    base_path='../Summaries_AlternationN/', params=params)
# Save all python files in current directory to script directory
files = glob.iglob(os.path.join('', '*.py'))
for file in files:
    if os.path.isfile(file):
        shutil.copy2(file, os.path.join(script_path, file))

# Save parameters
np.save(os.path.join(save_path, 'params'), dict(params))

# Create a logger to write log output to file
logger_sums = du.make_logger(run_path, 'summaries')
logger_envs = du.make_logger(run_path, 'env_details')
# Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
summary_writer = SummaryWriter(train_path)

if params.misc.use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="wm_em_slots",
        # track hyperparameters and run metadata
        config={x: params[x.split('.')[0]][x.split('.')[1]] for x in params.misc.org_rule},
        dir='../',
        name=run_name,
    )

# make instance of cscg em learner
model = _model_.AlternationN_Torch(params.model)
# put model to gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
msg = 'Device: ' + device.type
logger_sums.info(msg)
model = model.to(device)
# Make an ADAM optimizer
wd = params.train.weight_l2_reg_val if 'weight_l2' in params.train.which_costs else 0.0
if params.train.optimiser == 'adamW':
    optimizer = optim.AdamW(model.parameters(), lr=params.train.learning_rate_max, weight_decay=wd,
                            amsgrad=params.train.amsgrad)
elif params.train.optimiser == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=params.train.learning_rate_max, weight_decay=wd,
                           amsgrad=params.train.amsgrad)
else:
    raise ValueError('Optimiser type not implemented')

if params.data.load_data:
    ds_name, dataset_exists, _ = du.get_saved_data_id(params, '../Datasets_AlternationN/')
    try:
        ds_loaded = np.load('../Datasets_AlternationN/' + ds_name + '/dataset.npy', allow_pickle=True)
        ds = du.MyDataset(ds_loaded, batch_size=params.data.batch_size, shuffle=True)
    except FileNotFoundError:
        print('No dataset stored for this parameter configuration: ', ds_name)
        if device.type == 'cpu':
            ds = None
        else:
            raise ValueError('No dataset stored for this parameter configuration: ', ds_name)
    # dataset_loaded = torch.load('../Datasets_AlternationN/' + str(i) + '/dataset.pt')
    # ds = DataLoader(dataset_loaded, batch_size=params.data.batch_size, shuffle=True, num_workers=0)
else:
    ds = None

# initialise dictionary to contain environments and data info
train_dict = du.get_initial_data_dict(params.data, params.model.h_size)

msg = 'Training Started'
logger_sums.info(msg)
logger_envs.info(msg)
train_i = 0
debug_data = False
for train_i in range(params.train.train_iters):

    # INITIALISE ENVIRONMENT AND INPUT VARIABLES
    if train_i % params.misc.sum_int == 0 or train_i < 100:
        if sum(train_dict.env_steps == 0) > 0:
            msg = str(sum(train_dict.env_steps == 0)) + ' New Environments ' + str(train_i) + ' ' + str(
                train_i * params.data.seq_len)
            logger_envs.info(msg)

    # Get scaling parameters
    scalings = parameters.get_scaling_parameters(train_i, params.train)
    optimizer.lr = scalings.l_r

    data_start_time = time.time()
    # collect batch-specific environment data
    train_dict = du.data_step(train_dict, params.data, load=ds.next() if ds is not None else None)
    # convert all inputs to tensors
    inputs_torch = mu.inputs_2_torch(train_dict.inputs, scalings, device=device)
    if debug_data:
        print(train_i)
        continue

    # set all gradients to None
    # optimizer.zero_grad()
    for param in model.parameters():
        param.grad = None
    # forward pass
    forward_start_time = time.time()
    variables, re_input = model(inputs_torch, device=device)
    # collate inputs for model
    losses = _model_.compute_losses_torch(inputs_torch, variables, model, params.train, device=device,
                                          world_type=params.data.world_type)
    # backward pass
    backward_start_time = time.time()
    losses.train_loss.backward()

    # try to find nans / infs in the weightings
    if mu.is_any_nan_inf([mu.nested_isnan_inf(re_input), mu.nested_isnan_inf([x.data for x in model.parameters()]),
                          mu.nested_isnan_inf([x.grad for x in model.parameters()]),
                          mu.nested_isnan_inf(variables), mu.nested_isnan_inf(inputs_torch)]):
        print('NANS/INFS somewhere')
        breakpoint()

    # get gradient norms
    norms, total_norm = mu.gradient_norms(model)
    # clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
    # update model parameters
    optimizer.step()
    stop_time = time.time()

    # feeding in correct initial states
    re_input = mu.torch2numpy(re_input)
    train_dict.variables.hidden = re_input.hidden

    # Update logging info
    if train_i % params.misc.sum_int == 0 or train_i < 100:
        train_dict.curric_env.data_step_time = forward_start_time - data_start_time
        train_dict.curric_env.forward_step_time = backward_start_time - forward_start_time
        train_dict.curric_env.backward_step_time = stop_time - backward_start_time
        msg = 'Step {:.0f}, data time : {:.4f} , forward time {:.4f}, backward time {:.4f}' \
            .format(train_i, train_dict.curric_env.data_step_time, train_dict.curric_env.forward_step_time,
                    train_dict.curric_env.backward_step_time)
        logger_envs.info(msg)

    # Log training progress summary statistics
    if train_i % params.misc.sum_int == 0:
        msg = "train_i={:.2f}, total_steps={:.2f}".format(train_i, train_i * params.data.seq_len)
        logger_sums.info(msg)
        if train_i % (params.misc.sum_int * params.misc.mult_sum_metrics) == 0:
            metrics, train_dict.curric_env.metric_time = ca.compute_metrics(model, params, device=device,
                                                                            batch_size=300, train_i=train_i,
                                                                            non_lin=params.misc.non_lin)
            msg = "Metric time={:.2f}".format(train_dict.curric_env.metric_time)
            logger_sums.info(msg)
        else:
            metrics, train_dict.curric_env.metric_time = {}, None
        train_dict.curric_env.num_epochs = ds.num_epochs if ds is not None else None
        accuracies = mu.compute_accuracies_torch(inputs_torch, variables.pred, params.data)
        summaries = mu.make_summaries_torch(inputs_torch, losses, accuracies, scalings, variables,
                                            train_dict.curric_env, train_dict.inputs.seq_index,
                                            model, metrics, norms, params)
        for key_, val_ in summaries.items():
            if val_ is None:
                continue
            summary_writer.add_scalar(key_, val_, train_i)
        summary_writer.flush()
        if params.misc.use_wandb:
            wandb.log(summaries, step=train_i)

        losses = mu.torch2numpy(losses)
        msg = 'Losses: ' + ''.join(f'{key}={str(val)[:5]}, ' for key, val in losses.items() if 'unscaled' not in key)
        logger_sums.info(msg)
        msg = 'Accuracies: ' + ''.join(
            f'{key}={str(val)[:5]}, ' for key, val in accuracies.items() if 'post' not in key and 'second' not in key)
        logger_sums.info(msg)

    # Save model parameters which can be loaded later to analyse model
    if train_i % params.misc.save_interval == 0 and train_i > 0:
        start_time = time.time()
        # save model checkpoint
        # torch.save(model, model_path + '/rnn_' + str(train_i))
        # remove other models
        if params.misc.only_save_latest_model:
            files = glob.glob(model_path + '/*')
            for f in files:
                os.remove(f)
        torch.save(model.state_dict(), model_path + '/rnn_' + str(train_i))
        logger_sums.info("save data time {:.2f}, train_i={:.2f}, total_steps={:.2f}".format(
            time.time() - start_time, train_i, train_i * params.data.seq_len))

print('Finished training')

# save final copy of model
# du.save_model_outputs(model, mu, train_i, save_path, params)
torch.save(model.state_dict(), model_path + '/rnn_' + str(train_i))
if params.misc.use_wandb:
    wandb.finish()
