#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import os
import sys
import shutil
import datetime
import logging
import torch
import time
import environments
import parameters
from typing import Dict, Any
import hashlib
import json

import numpy as np
import copy as cp
import model_utils as mu

from distutils.dir_util import copy_tree
from torch.utils.data import Dataset
from deepdiff import DeepDiff


# from multiprocessing import Pool
# from multiprocessing.pool import ThreadPool
# from line_profiler_pycharm import profile


class MyDataset(Dataset):
    def __init__(self, data, batch_size=16, shuffle=True):
        self.data = np.random.permutation(data) if shuffle else data
        self.iteration = 0
        self.num_epochs = 0
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

    def next(self):
        start, stop = self.iteration * self.batch_size, (self.iteration + 1) * self.batch_size
        if stop > len(self.data):
            self.iteration = 0
            self.num_epochs += 1
            if self.shuffle:
                self.data = np.random.permutation(self.data)
            start, stop = self.iteration * self.batch_size, (self.iteration + 1) * self.batch_size

        batch = self.data[start:stop]
        self.iteration += 1
        return batch


def make_directories(base_path='../Summaries/', params=None):
    """
    Creates directories for storing data during a model training run
    """

    if params is not None:
        try:
            org_rule = [(x.split('.')[0], x.split('.')[-1]) for x in params.misc.org_rule]
            name = [str(params[a][b]) for (a, b) in org_rule]
            for i, n in enumerate(name):
                n = n.replace(',', '')
                n = n.replace('.', '')
                n = n.replace(' ', '')
                if n == 'loop':
                    n = n + '_' + params['data']['behaviour_type']
                name[i] = n
            name = ' ' + ' '.join(name)
        except KeyError:
            name = ''
    else:
        name = ''

    # Get current date for saving folder
    date = datetime.datetime.today().strftime('%Y-%m-%d')
    # Initialise the run and dir_check to create a new run folder within the current date
    run = 0
    dir_check = True
    # Initialise all paths
    train_path, model_path, save_path, script_path, run_path, envs_path = None, None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths (allowed a max of 10000 runs per day)
        run_name = date + name + '/run' + ('000' + str(run))[-4:]
        run_path = base_path + run_name + '/'
        train_path = run_path + 'train'
        model_path = run_path + 'model'
        save_path = run_path + 'save'
        script_path = run_path + 'script'
        envs_path = script_path + '/envs'
        run += 1
        # And once a path doesn't exist yet: create new folders
        if not os.path.exists(train_path) and not os.path.exists(model_path) and not os.path.exists(save_path):
            try:
                os.makedirs(train_path)
                os.makedirs(model_path)
                os.makedirs(save_path)
                os.makedirs(script_path)
                os.makedirs(envs_path)
                dir_check = False
            except FileExistsError:
                # often multiple jobs run at same time and get fudged here, so add this catch statement
                pass
    if run > 10000:
        raise ValueError("While loop for making directory was going on forever")

    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path, run_name


def set_directories(date, run, base_path='../Summaries/'):
    """
    Returns directories for storing data during a model training run from a given previous training run
    """

    # Initialise all paths
    run_path = base_path + date + '/run' + str(run) + '/'
    train_path = run_path + 'train'
    model_path = run_path + 'model'
    save_path = run_path + 'save'
    script_path = run_path + 'script'
    envs_path = script_path + '/envs'
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path


def make_logger(run_path, name):
    """
    Creates logger so output during training can be stored to file in a consistent way
    """

    # Create new logger    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Remove anly existing handlers so you don't output to old files, or to new files twice
    # - important when resuming training existing model
    logger.handlers = []
    # Create a file handler, but only if the handler does
    handler = logging.FileHandler(run_path + name + '.log')
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    # Return the logger object
    return logger


def save_params(pars, save_path, script_path):
    np.save(save_path + '/params', dict(pars))
    copy_tree('./', script_path)
    return


def get_next_batch(position, action, edge_visits, pars):
    # s_visited is for each bptt, saying whether each the state at current timestep has been visited before
    s_visited = np.ones((pars.seq_len, pars.batch_size), dtype=np.int32)
    inference_opportunity = np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32)

    for seq in range(pars.seq_len):
        pos = position[seq, :].astype(int)
        current_node_visits = np.sum(edge_visits, axis=2)[np.arange(pars.batch_size), pos]
        current_edge_visits = edge_visits[np.arange(pars.batch_size), pos, action[seq]]
        # have I visited this position before - 1 if yes, 0 if no
        s_visited[seq, :] = (current_node_visits > 0).astype(int)
        # inference available if arrive at old state from a new direction!
        inference_opportunity[seq, :] = (np.logical_and(current_edge_visits == 0, current_node_visits > 0)).astype(int)
        # add position to places I've been
        edge_visits[np.arange(pars.batch_size), pos, action[seq]] += 1

    new_data = (edge_visits, s_visited, inference_opportunity)

    return new_data


def initialise_variables(env_steps, data_dict):
    for env, env_step in enumerate(env_steps):
        # only do if just entered environment
        if env_step > 0:
            continue

        data_dict.hidden[env, ...] = 0
        data_dict.edge_visits[env, ...] = 0
    return data_dict


def prepare_cell_timeseries(prev_data, data):
    if prev_data is None:
        return cp.deepcopy(data)
    else:
        for key, val in data.items():
            prev_data[key] = np.concatenate([prev_data[key], data[key]], axis=0)
        return prev_data


def prepare_input(data_dict, pars, start_i=None):
    """
    Select section of walk sequences that gets fed into model, and prepare model input dictionary
    """
    # select right bit of data to send to model
    i1 = data_dict.env_steps * pars.seq_len if start_i is None else start_i
    i2 = i1 + pars.seq_len
    for batch in range(pars.batch_size):
        for key, val in data_dict.walk_data.items():
            if val[batch] is None:
                continue
            data_dict.bptt_data[key][:, batch, ...] = val[batch][i1[batch]:i2[batch], ...]

    # convert positions to sensory observations, and get node/edge visit info
    new_data = get_next_batch(data_dict.bptt_data.position, data_dict.bptt_data.action,
                              data_dict.variables.edge_visits, pars)
    edge_visits, s_visited, inference_opportunity = new_data

    # model input data
    data_dict.inputs = cp.deepcopy(data_dict.bptt_data)
    data_dict.inputs.hidden = data_dict.variables.hidden
    data_dict.inputs.seq_index = np.array(data_dict.env_steps, dtype=np.int32)
    data_dict.inputs.s_visited = s_visited

    data_dict.variables.edge_visits = edge_visits
    data_dict.variables.inference_opportunity = inference_opportunity

    # update env_steps
    data_dict.env_steps += 1
    # new environment if finished all data from walk
    data_dict.env_steps[i2 >= [len(x) for x in data_dict.walk_data.position]] = 0

    return data_dict


def get_initial_data_dict(pars, h_size):
    # prepare_environment_data
    data_dict = mu.DotDict({'env_steps': np.zeros(pars.batch_size).astype(int),
                            'curric_env':
                                {'envs': [None for _ in range(pars.batch_size)],
                                 'walk_len': np.zeros(pars.batch_size).astype(int),
                                 'states_mat': [0 for _ in range(pars.batch_size)],
                                 'adjs': [0 for _ in range(pars.batch_size)],
                                 'trans': [0 for _ in range(pars.batch_size)],
                                 },
                            'variables':
                                {'hidden': np.zeros((pars.batch_size, h_size)),
                                 'edge_visits': np.zeros((pars.batch_size, pars.max_states, pars.env.n_actions)),
                                 'start_state': np.zeros(pars.batch_size),
                                 },
                            'walk_data':
                                {'position': [None for _ in range(pars.batch_size)],
                                 'action': [None for _ in range(pars.batch_size)],
                                 'reward': [None for _ in range(pars.batch_size)],
                                 'exploration': [None for _ in range(pars.batch_size)],
                                 'steps_between_rewards': [None for _ in range(pars.batch_size)],
                                 'phase_velocity': [None for _ in range(pars.batch_size)],
                                 'phase': [None for _ in range(pars.batch_size)],
                                 'velocity': [None for _ in range(pars.batch_size)],
                                 'travelling': [None for _ in range(pars.batch_size)],
                                 'observation': [None for _ in range(pars.batch_size)],
                                 'goal_position': [None for _ in range(pars.batch_size)],
                                 'goal_observation': [None for _ in range(pars.batch_size)],
                                 'target_o': [None for _ in range(pars.batch_size)],
                                 'chunk_action': [None for _ in range(pars.batch_size)],
                                 },
                            'bptt_data':
                                {'position': np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'action': np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'reward': np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'exploration': np.ones((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'steps_between_rewards': np.ones((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'phase_velocity': np.zeros((pars.seq_len, pars.batch_size), dtype=np.float32),
                                 'phase': np.zeros((pars.seq_len, pars.batch_size), dtype=np.float32),
                                 'velocity': np.zeros((pars.seq_len, pars.batch_size, pars.env.dim_space),
                                                      dtype=np.float32),
                                 'travelling': np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'observation': np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'goal_position': np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'goal_observation': np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'target_o': np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 'chunk_action': np.zeros((pars.seq_len, pars.batch_size), dtype=np.int32),
                                 },
                            })
    return data_dict


def initialise_environments(curric_env, env_steps, pars, test=False, load=None, algebra=None):
    if load is None:
        for b, (env, env_step) in enumerate(zip(curric_env.envs, env_steps)):
            # only do if just entered environment
            if env_step > 0:
                continue

            if pars.world_type in ['rectangle']:
                curric_env.envs[b] = environments.Rectangle(pars, pars.env.widths[b], pars.env.heights[b])
            elif pars.world_type in ['rectangle_chunk']:
                curric_env.envs[b] = environments.RectangleChunk(pars, pars.env.widths[b], pars.env.heights[b])
            elif pars.world_type in ['rectangle_behave']:
                curric_env.envs[b] = environments.RectangleBehave(pars, pars.env.widths[b], pars.env.heights[b])
            elif pars.world_type in ['Basu2021']:
                curric_env.envs[b] = environments.RectangleRewards(pars, pars.env.widths[b], pars.env.heights[b])
            elif pars.world_type == 'NBack':
                curric_env.envs[b] = environments.NBack(pars, pars.env.widths[b], pars.env.heights[b])
            elif pars.world_type == 'loop':
                curric_env.envs[b] = environments.Loop(pars, pars.env.widths[b], pars.env.heights[b])
            elif pars.world_type == 'loop_chunk':
                curric_env.envs[b] = environments.LoopChunk(pars, pars.env.widths[b], pars.env.heights[b])
            elif pars.world_type in ['loop_delay', 'loop_same_delay']:
                curric_env.envs[b] = environments.LoopDelay(pars, pars.env.widths[b], pars.env.heights[b])
            elif pars.world_type == 'Panichello2021':
                curric_env.envs[b] = environments.Panichello2021(pars, pars.env.widths[b], pars.env.heights[b])
            elif pars.world_type == 'Xie2022':
                curric_env.envs[b] = environments.Xie2022(pars, pars.env.widths[b], pars.env.heights[b])

            curric_env.envs[b].world()
            curric_env.envs[b].state_data()
    else:
        for key in load[0].curric_env.keys():
            curric_env[key] = [x.curric_env[key] for x in load]

    for b, (env, env_step) in enumerate(zip(curric_env.envs, env_steps)):
        # only do if just entered environment
        if env_step > 0:
            continue
        if load is None:
            curric_env.envs[b].walk_len = pars.seq_len
        curric_env.walk_len[b] = pars.seq_len

    if algebra is not None:
        # replace state_data with previous things for algebra
        # Edit observations for algebra
        index_, observation_to_remove, observation_to_add = None, None, None
        for b, (_, _) in enumerate(zip(curric_env.envs, env_steps)):
            # i - (i+1) + (i+2) = (i+3)
            if b % 4 == 0:
                # choose which index to change
                index_ = np.random.randint(len(curric_env.envs[b].states_mat))
                # get that observation
                observation_to_remove = curric_env.envs[b].states_mat[index_]
            elif b % 4 == 1:
                # new observation to add
                observation_to_add = curric_env.envs[b].states_mat[index_]
                curric_env.envs[b].states_mat[index_] = cp.deepcopy(observation_to_remove)
            elif b % 4 == 2:
                curric_env.envs[b].states_mat = cp.deepcopy(curric_env.envs[b - 1].states_mat)
                curric_env.envs[b].states_mat[index_] = cp.deepcopy(observation_to_add)
            elif b % 4 == 3:
                curric_env.envs[b].states_mat = cp.deepcopy(curric_env.envs[b - 3].states_mat)
                curric_env.envs[b].states_mat[index_] = cp.deepcopy(observation_to_add)

    return curric_env


def work(instance):
    return instance.walk()


# @profile
def get_walk_data_class(data_dict, envs, env_steps, algebra=None):
    """
    pool = ThreadPool(4)
    time_ = time.time()
    with pool:
        results = pool.map(work, envs.envs, chunksize=30)
    for b, walk_data in enumerate(results):
        for key, val in walk_data.items():
            data_dict[key][b] = val
    pool.close()
    pool.join()
    print('pool', time.time() - time_)
    """
    # time_ = time.time()
    for b, (env, env_step) in enumerate(zip(envs.envs, env_steps)):
        # only do if just entered environment
        if env_step > 0:
            continue

        walk_data = env.walk()
        for key, val in walk_data.items():
            data_dict[key][b] = val
    # print('sequential', time.time() - time_)

    if algebra == 'seq':
        for b, (env, _) in enumerate(zip(envs.envs, env_steps)):
            # i - (i+1) + (i+2) = (i+3)
            if b % 4 == 0:
                pass
            else:
                data_dict['position'][b] = data_dict['position'][b - 1]
                data_dict['velocity'][b] = data_dict['velocity'][b - 1]
                # set observations to be correct
                for i, p in enumerate(data_dict['position'][b]):
                    data_dict['observation'][b][i] = env.states_mat[p]

    return data_dict


def add_intermediary_steps(data_dict, envs, env_steps, pars):
    for b, (env, env_step) in enumerate(zip(envs.envs, env_steps)):
        # only do if just entered environment
        if env_step > 0:
            continue

        walk_len = envs.walk_len[b]
        for key, value in data_dict.items():
            if key in ['travelling'] or data_dict[key][b] is None:
                continue
            # these are all seq_len x XXX (most just seq_len)
            shape = list(data_dict[key][b].shape)
            shape[0] *= (pars.intermediate_steps + 1)
            new = np.zeros(shape, dtype=data_dict[key][b].dtype)
            new[..., ::(pars.intermediate_steps + 1)] = data_dict[key][b]
            data_dict[key][b] = new[..., :walk_len]

        # add 'travelling' data - 1 if 'travelling', i.e. process of going between states. 0 if at real state.
        # if pars.world_type == 'Basu2021' and pars.only_train_on_rewards:
        #    data_dict.travelling[b] = 1 - data_dict.reward[b]  # 'only make predictions on reward states
        # else:
        travelling = np.ones(walk_len * (pars.intermediate_steps + 1))
        travelling[::(pars.intermediate_steps + 1)] = 0
        data_dict.travelling[b] = travelling[:walk_len]

    return data_dict


# @profile
def data_step(data, pars, test=False, load=None, algebra=None):
    """
    could do env step loop here, with curriculum etc only for one env at a time
    """
    # make environments
    data.curric_env = initialise_environments(data.curric_env, data.env_steps, pars, test=test, load=load,
                                              algebra=algebra)
    # initialise all other variables
    data.variables = initialise_variables(data.env_steps, data.variables)
    if load is None:
        # Collect full sequence of data
        data.walk_data = get_walk_data_class(data.walk_data, data.curric_env, data.env_steps, algebra=algebra)
        # add intermediary steps
        data.walk_data = add_intermediary_steps(data.walk_data, data.curric_env, data.env_steps, pars)
    else:
        for key in data.walk_data.keys():
            try:
                data.walk_data[key] = [x.walk_data[key] for x in load]
            except IndexError:
                pass
    # Select section of walk sequences that gets fed into model, and prepare model input dictionary
    data_dict = prepare_input(data, pars)

    return data_dict


def save_model_outputs(model, mu_, train_i, iter_path, pars, device='cpu'):
    """
    Takes a model and collects cell and environment timeseries from a forward pass
    """
    # Initialise timeseries data to collect
    variables_test, timeseries = None, None
    # Initialise model input data
    test_dict = get_initial_data_dict(pars.data, pars.model.h_size)
    # Run forward pass
    ii, data_continue = 0, True
    while data_continue:
        # Update input
        test_dict = data_step(test_dict, pars.data, test=True)
        scalings = parameters.get_scaling_parameters(train_i, pars.train)
        inputs_torch = mu.inputs_2_torch(test_dict.inputs, scalings, device=device)
        # Do model forward pass step
        with torch.no_grad():
            variables_test, re_input_test = model(inputs_torch)
        re_input_test = mu_.torch2numpy(re_input_test)
        test_dict.variables.hidden = re_input_test.hidden

        # Collect environment step data: position and observation
        hidden = mu_.torch2numpy(variables_test.hidden)
        # Update timeseries
        timeseries = prepare_cell_timeseries(timeseries, hidden)

        ii += 1
        print(str(ii) + '/' + str(int(len(test_dict.walk_data.position[0]) / pars.data.seq_len)), end=' ')
        if sum(test_dict.env_steps) == 0:
            data_continue = False

    # save all final variables
    if not os.path.exists(iter_path):
        os.makedirs(iter_path)

    # save all data
    np.save(iter_path + '/final_variables_' + str(train_i), mu_.DotDict.to_dict(variables_test), allow_pickle=True)
    # Save all timeseries to file
    np.save(iter_path + '/timeseries_' + str(train_i), mu_.DotDict.to_dict(timeseries))

    # Convert test_dict, which is DotDicts, to a normal python dictionary - don't want any DotDicts remaining
    final_dict = mu_.DotDict.to_dict(test_dict)

    # convert class params to dict
    for i, env in enumerate(final_dict['curric_env']['envs']):
        final_dict['curric_env']['envs'][i].par = mu_.DotDict.to_dict(env.par)

    # Save final test_dict to file, which contains all environment info
    np.save(iter_path + '/final_dict_' + str(train_i), final_dict, allow_pickle=True)

    return


def new2stored_memories(memories_dict_, pars):
    """
    Takes 'new' memories and puts them into 'stored' memories.
    Only keeps memories around that have non-zero weighting

    :param memories_dict_:
    :param pars:
    :return:
    """

    memories_dict = mu.DotDict(cp.deepcopy(mu.DotDict.to_dict(memories_dict_)))

    for b in range(pars.batch_size):

        # remove memories that were deleted -  - i.e. with zero in 'weighting' (min val is zero I hope)
        indices = np.where(memories_dict.stored.in_use[b, :] == pars.prune_mems_corr_threshold)[0]
        memories_dict.stored.x[b, :, indices] = 0.0
        memories_dict.stored.g[b, :, indices] = 0.0
        # re-order mems so that all memories to keep are at the 'front'
        idx = np.argsort(np.abs(memories_dict.stored.in_use[b, :]))[::-1]
        memories_dict.stored.x[b, :, :] = memories_dict.stored.x[b, :, idx].T
        memories_dict.stored.g[b, :, :] = memories_dict.stored.g[b, :, idx].T
        memories_dict.stored.in_use[b, :] = memories_dict.stored.in_use[b, idx]

        # Remove 'new' memories that were deleted - i.e. with zero in 'weighting' (min val is zero I hope)
        indices = np.where(memories_dict.new.in_use[b, :] != pars.prune_mems_corr_threshold)[0]
        n = len(indices)

        if n > 0:
            memories_dict.stored.x[b, :, n:] = memories_dict.stored.x[b, :, :-n]
            memories_dict.stored.x[b, :, :n] = memories_dict.new.x[b, :, indices].T

            memories_dict.stored.g[b, :, n:] = memories_dict.stored.g[b, :, :-n]
            memories_dict.stored.g[b, :, :n] = memories_dict.new.g[b, :, indices].T

            memories_dict.stored.in_use[b, n:] = memories_dict.stored.in_use[b, :-n]
            memories_dict.stored.in_use[b, :n] = memories_dict.new.in_use[b, indices]

    # return stored memories
    new_dict = mu.DotDict({'x': memories_dict.stored.x,
                           'g': memories_dict.stored.g,
                           'in_use': memories_dict.stored.in_use,
                           })

    return new_dict


def data_dict_2_batch(data_dict, batch_size):
    # go to leaf directories which is bacthed, and make a batch of dicts instead
    return [mu.DotDict(nested_dict_batch(data_dict, b)) for b in range(batch_size)]


def nested_dict_batch(x, batch):
    if isinstance(x, mu.DotDict) or isinstance(x, dict):
        return {key: nested_dict_batch(value, batch) for key, value in x.items()}
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return x[batch]
    else:
        return x


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_saved_data_id(params, path, delete=False):
    dataset_exists = False
    deleted_dataset = 0
    par_new = cp.deepcopy(params.data)
    par_new.pop('batch_size', None)
    par_new.env.widths = par_new.env.widths[0]
    par_new.env.heights = par_new.env.heights[0]
    if not os.path.exists(path):
        os.mkdir(path)

    for dataset in os.listdir(path):
        if dataset == '.DS_Store':
            continue
        try:
            par_old = parameters.load_params(path + dataset).data
            par_old.pop('batch_size', None)
            par_old.env.widths = par_old.env.widths[0]
            par_old.env.heights = par_old.env.heights[0]
            if DeepDiff(par_new, par_old) == {}:
                # Check if matched dataset is corrupted or not
                print('Dataset already exists: ', dataset)
                try:
                    if os.stat(path + dataset + '/dataset.npy').st_size / (1024 * 1024) < 1.0:
                        print('Dataset has zero size. Likely not properly created.')
                        if delete:
                            print('Removing dataset folder: ', dataset)
                            shutil.rmtree(path + dataset)
                            # os.rmdir(path + dataset)
                            deleted_dataset += 1
                            break
                        else:
                            raise ValueError('Dataset has zero size. Likely not properly created. ' +
                                             'Set delete on if you want it deleted')
                    else:
                        print('Dataset is reasonable size, assuming it was properly created.')
                        dataset_exists = True
                        break
                except FileNotFoundError:
                    # Check if dataset in process of being created
                    if delete:
                        print('Dataset not yet created. Either being created, or creation process crashed.')
                        print('Checking if env_details recently updated.')
                        try:
                            if time.time() - os.path.getmtime(path + dataset + '/env_details.log') > 300:
                                print('env_details over 300 seconds not updated. Assume creation process crashed.')
                                print('Removing dataset folder: ', dataset)
                                shutil.rmtree(path + dataset)
                                # os.rmdir(path + dataset)
                                deleted_dataset += 1
                                break
                            else:
                                raise ValueError('Env_details updated less that 300 seconds ago. Assume still running')
                        except FileNotFoundError:
                            print('Env_details does not exist. Assume creation process crashed.')
                            print('Removing dataset folder: ', dataset)
                            shutil.rmtree(path + dataset)
                            # os.rmdir(path + dataset)
                            deleted_dataset += 1
                            break
                    else:
                        raise ValueError('Dataset not yet created. Either being created, or creation process crashed.' +
                                         'Set delete on if you want it deleted')
            else:
                # Dataset id different from this saved dataset
                pass
        except FileNotFoundError:
            # Couldn't load params, therefore folder is empty
            # remove folder
            print('Dataset folder empty. Removing dataset folder: ', dataset)
            shutil.rmtree(path + dataset)
            # os.rmdir(path + dataset)
            deleted_dataset += 1

    par_new = {key: int(val) if isinstance(val, np.int64) else val for key, val in par_new.items()}
    name = dict_hash(par_new)
    # print(name, name == 'c89975b389a21a9e324c72b036f15400')
    return name, dataset_exists, deleted_dataset
