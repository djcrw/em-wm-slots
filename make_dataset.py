#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import parameters
import os
import time
import data_utils as du
import numpy as np

# Initialise hyper-parameters for model
params = parameters.default_params()

if not params.data.load_data:
    raise ValueError('Load data setting must be True')

# Create directories for storing all information about the current run
path = '../Datasets_AlternationN/'
if not os.path.exists(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

# iterate through Dataset folder, find setting with exact same data params
ds_name, dataset_exists, deleted_dataset = du.get_saved_data_id(params, path, delete=True)
if dataset_exists:
    raise ValueError('Dataset already exists: ', ds_name)

# if no dataset found, make new folder
new_path = path + ds_name
if not os.path.exists(new_path):
    print('Making new dataset folder: ', ds_name)
    os.makedirs(new_path)
    np.save(os.path.join(new_path, 'params'), dict(params))

# Create a logger to write log output to file
logger_envs = du.make_logger(new_path + '/', 'env_details')
msg = 'Dataset already exist: ' + str(dataset_exists) + ', Deleted datasets: ' + str(deleted_dataset)
logger_envs.info(msg)

# initialise dictionary to contain environments and data info
msg = 'Started'
logger_envs.info(msg)
list_of_data = []
for train_i in range(params.data.training_data_save):  # params.train.train_iters):
    train_dict = du.get_initial_data_dict(params.data, params.model.h_size)

    # INITIALISE ENVIRONMENT AND INPUT VARIABLES
    if sum(train_dict.env_steps == 0) > 0:
        msg = str(sum(train_dict.env_steps == 0)) + ' New Environments ' + str(train_i) + ' ' + str(
            train_i * params.data.seq_len)
        logger_envs.info(msg)

    data_start_time = time.time()
    # collect batch-specific environment data
    train_dict = du.data_step(train_dict, params.data)
    # remove things we don't need to save
    train_dict.pop('variables', None)
    train_dict.pop('bptt_data', None)
    train_dict.pop('inputs', None)
    train_dict.pop('env_steps', None)
    train_dict.curric_env.pop('envs', None)

    list_of_data = list_of_data + du.data_dict_2_batch(train_dict, params.data.batch_size)

    # convert all inputs to tensors
    # inputs_torch = mu.inputs_2_torch(train_dict.inputs, scalings, device=device)

np.save(new_path + '/dataset.npy', list_of_data)
print('Dataset fully made and saved: ', ds_name)
# dataset = du.MyDataset(list_of_data)
# torch.save(dataset, new_path + '/dataset.pt')  # SAVE THIS AS NUMPY -> THEN USE ALL SAME STUFF AS BEFORE...
# dataloader = DataLoader(dataset, batch_size=5)
