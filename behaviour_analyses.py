#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

from model_utils import DotDict
import data_utils
import importlib.util
import os
import scipy.stats as stats
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import parameters
import torch
import plotting_functions

fontsize = 25
linewidth = 4
labelsize = 20


def save_trained_outputs(date, run, index, use_old_scripts=True, base_path='../Summaries/', force_overwrite=False,
                         n_envs_save=6, new_batch_size=None):
    """
    Load a trained model from a previous training run and save outputs
    """
    # Get directories for the requested run
    run_path, train_path, model_path, save_path, script_path, envs_path = \
        data_utils.set_directories(date, run, base_path=base_path)
    # Load model from file
    model, params = get_model(model_path, script_path, save_path, index, use_old_scripts=True,
                              new_batch_size=new_batch_size)

    par_new = parameters.default_params(batch_size=new_batch_size)
    for key_1 in par_new.keys():
        for key_2 in par_new[key_1].keys():
            try:
                params[key_1][key_2]
            except (KeyError, AttributeError) as e:
                params[key_1][key_2] = par_new[key_1][key_2]

    if new_batch_size is not None:
        params.data.batch_size = new_batch_size

    # set n_envs_save to be high
    params.misc.n_envs_save = n_envs_save
    # Make sure there is a trained model for the requested index (training iteration)
    if model is not None:
        # If the output directory already exists: only proceed if overwriting is desired
        iter_path = save_path + '/iter_' + str(index)
        if os.path.exists(iter_path) and os.path.isdir(iter_path):
            files_exist = os.listdir(iter_path)
            if not files_exist and force_overwrite:
                print('Running forward pass to collect data')
            else:
                print('Not running forward pass: ' + iter_path + ' already exists')
                if not files_exist:
                    print('But no files exist!')
                return model, params

        # Load data_utils from stored scripts of trained model (for compatibility) or current (for up-to-date)
        spec_data_utils = importlib.util.spec_from_file_location("data_utils", script_path + '/data_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("data_utils", 'data_utils.py')
        stored_data_utils = importlib.util.module_from_spec(spec_data_utils)
        spec_data_utils.loader.exec_module(stored_data_utils)
        # Load model_utils from stored scripts of trained model
        spec_model_utils = importlib.util.spec_from_file_location("model_utils", script_path + '/model_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("model_utils", 'model_utils.py')
        stored_model_utils = importlib.util.module_from_spec(spec_model_utils)
        spec_model_utils.loader.exec_module(stored_model_utils)
        # Create output folder
        if not os.path.exists(iter_path):
            os.makedirs(iter_path)
        # Run forward pass and collect model outputs, then save model outputs to save forlder
        stored_data_utils.save_model_outputs(model, stored_model_utils, index, iter_path, params)
    return model, params


def get_model(model_path, script_path, save_path, index, use_old_scripts=True, new_batch_size=None):
    """
    Load a trained model from a previous training run and save outputs
    """
    # Make sure there is a trained model for the requested index (training iteration)
    if os.path.isfile(model_path + '/rnn_' + str(index)):
        model_path = model_path + '/rnn_' + str(index)
    else:
        print('Error: no trained model found for ' + model_path + '/rnn_' + str(index))
        # Return None to indicate error
        return None, None
    try:
        # Load model module from stored scripts of trained model
        spec_model = importlib.util.spec_from_file_location("rnn", script_path + '/model.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("rnn", 'model.py')
        stored_model = importlib.util.module_from_spec(spec_model)
        spec_model.loader.exec_module(stored_model)
        # Load data_utils from stored scripts of trained model
        spec_data_utils = importlib.util.spec_from_file_location("data_utils", script_path + '/data_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("data_utils", 'data_utils.py')
        stored_data_utils = importlib.util.module_from_spec(spec_data_utils)
        spec_data_utils.loader.exec_module(stored_data_utils)
        # Load model_utils from stored scripts of trained model
        spec_model_utils = importlib.util.spec_from_file_location("model_utils", script_path + '/model_utils.py') \
            if use_old_scripts else importlib.util.spec_from_file_location("model_utils", 'model_utils.py')
        stored_model_utils = importlib.util.module_from_spec(spec_model_utils)
        spec_model_utils.loader.exec_module(stored_model_utils)
        # Load the parameters of the model
        params = parameters.load_params(save_path)
        if new_batch_size is not None:
            spec_parameters = importlib.util.spec_from_file_location("rnn", script_path + '/parameters.py')
            stored_parameters = importlib.util.module_from_spec(spec_parameters)
            spec_parameters.loader.exec_module(stored_parameters)
            params = stored_parameters.default_params(batch_size=new_batch_size)
        try:
            print('Attempting to load from state_dict')
            # Create a new model with the loaded parameters
            model = stored_model.AlternationN_Torch(params.model)
            # Load the model weights after training
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        except:
            print('Attempting to load from state_dict = failed')
            print('Attempting to load full model')
            model = torch.load(model_path, map_location=torch.device('cpu'))

        model.eval()
        # Return loaded and trained model
        return model, params
    except ModuleNotFoundError:
        return None, parameters.load_params(save_path)
