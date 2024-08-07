#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
import environments
import os
import gzip
from model_utils import DotDict as Dd
from itertools import cycle, islice


def default_params(batch_size=None, h_size=None, width=None, height=None, seq_len_mult=7):
    data = Dd()
    model = Dd()
    train = Dd()
    misc = Dd()

    """
    ----------------------------------------------------------------
    ENVIRONMENT / DATA
    ----------------------------------------------------------------
    """
    # 'rectangle', 'loop', 'loop_delay', 'line', 'NBack', 'Basu2021', 'Panichello2021', 'Xie2022'
    # 'loop_same_delay'
    data.world_type = "Panichello2021"
    data.batch_size = 128 if not batch_size else batch_size
    data.o_size = 10
    data = get_env_params(data, width, height=height, seq_len_mult=seq_len_mult)
    data.load_data = False  # set True if you have run make_datesets...
    data.training_data_save = 5000
    data.intermediate_steps = 0
    data.sample_observation_without_replacement = True
    if data.sample_observation_without_replacement:
        if data.world_type in ['loop_delay', 'loop_same_delay']:
            assert (data.o_size >= data.env.widths[0])
        else:
            assert (data.o_size >= data.max_states)

    """
    ----------------------------------------------------------------
    MISC
    ----------------------------------------------------------------
    """
    # only save date from first X of batch
    misc.n_envs_save = 16
    # num gradient updates between summaries
    misc.sum_int = 200
    misc.mult_sum_metrics = 20
    misc.non_lin = False
    # number of gradient steps between saving model
    misc.save_interval = 10000
    misc.only_save_latest_model = True
    misc.log_weight_stats = False
    misc.log_var_stats = False
    misc.use_wandb = True if 'Dropbox' not in os.getcwd() else False
    # how to organise runs
    misc.org_rule = ['data.world_type', 'model.p_size', 'model.model_type', 'model.h_size', 'model.hidden_act',
                     'model.transition_type', 'model.transition_init', 'model.norm_pi_to_pred',
                     'train.hidden_l2_pen', 'train.weight_l2_reg_val', 'train.lh_kl_val', 'train.amsgrad',
                     'train.train_iters', 'train.two_dim_output', 'data.o_size',
                     'data.sample_observation_without_replacement']
    """
    ----------------------------------------------------------------
    MODEL
    ----------------------------------------------------------------
    """
    model.use_velocity = False
    if data.world_type in ['rectangle']:
        model.use_velocity = True
    elif data.world_type in ['loop']:
        if data.behaviour_type in ['random']:
            model.use_velocity = True

    # model variants:
    model.model_type = "WM"
    model.external_memory = True
    if model.model_type == 'WM':
        model.external_memory = False
    elif model.model_type == 'EM':
        pass
    else:
        raise ValueError('Incorrect model type specified')

    # 'conventional_rnn', 'group', 'bio_rnn_add', 'rnn_add',
    model.transition_type = "conventional_rnn"
    # 'leaky_relu_0.00' # 'relu', 'leaky_relu_X.XX', 'none', 'tanh'
    model.hidden_act = "relu"
    if model.transition_type == 'group' and model.hidden_act == 'tanh':
        model.hidden_act = 'none'
    model.hidden_thresh = 10.0
    model.hidden_thresh_alpha = 0.05

    # dimensions
    model.h_size = 128
    if h_size is not None:
        model.h_size = h_size
    model.key_size = 64
    model.value_size = model.h_size
    model.o_size = data.o_size
    model.n_rewards = data.n_rewards
    model.v_size = data.env.dim_space
    try:
        model.n_chunk_actions = data.n_chunk_actions
    except:
        model.n_chunk_actions = None
    model.d_mixed_size = 16
    model.bio_rnn_h_mult = 16 if model.transition_type in ['bio_rnn_add'] else 1
    model.embed_size = model.h_size
    model.norm_pi_to_pred = True

    # initialisations
    model.linear_std = 0.10
    model.embedding_std = 0.06
    model.hidden_init_std = 0.06
    model.transition_init = "trunc_norm"
    if model.transition_type == 'group':
        model.transition_init = 'trunc_norm'
    # RNN transition
    model.add_identity = True
    if model.transition_init == 'orthogonal':
        model.add_identity = False
    if model.transition_type == 'group':
        model.add_identity = True

    model.use_chunk_action = True if data.world_type in ['loop_chunk', 'rectangle_chunk'] else False
    # inputs and outputs
    if data.world_type in ['Basu2021', 'loop_delay', 'loop_same_delay']:
        model.embedding_inputs = ['observation', 'reward']
        model.to_predict = ['target_o']
    elif data.world_type in ['NBack', 'rectangle', 'rectangle_behave', 'Panichello2021', 'Xie2022']:
        model.embedding_inputs = ['observation']
        model.to_predict = ['target_o']
    else:
        model.embedding_inputs = ['observation']
        model.to_predict = ['target_o']


    """
    ----------------------------------------------------------------
    TRAINING
    ----------------------------------------------------------------
    """
    train.optimiser = 'adam'
    train.amsgrad = True
    train.train_iters = 500001
    train.train_on_visited_states_only = True
    train.learning_rate_max = 0.8e-3
    train.learning_rate_min = 0.2e-3
    train.which_costs = model.to_predict
    train.which_costs += ['weight_l2', 'hidden_kl', 'hidden_l2']
    # regularisation values
    train.lh_kl_val = 0.00
    train.hidden_l2_pen = 0.00
    train.weight_l2_reg_val = 5e-8
    # annealing learning rate
    train.l_r_decay_steps = 4000
    train.l_r_decay_rate = 0.5
    # for PFC tasks
    train.two_dim_output = False

    # dummy_line_for_repeat0
    return Dd({'data': data,
               'misc': misc,
               'model': model,
               'train': train})


def get_env_params(par, width, height, seq_len_mult):
    par.n_rewards = 4
    if par.world_type in ['rectangle']:
        width, height = 2, 2
        par.seq_len = seq_len_mult * width * height
        par.torus = True

        par_env = Dd({'stay_still': False,
                      'bias_type': 'none',  # 'angle',
                      'direc_bias': 0.25,
                      'angle_bias_change': 0.4,
                      'widths': [width] * par.batch_size,
                      'heights': [height] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right', 'stay still'],
                      'velocities': [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]],
                      'dim_space': 2,
                      'correct_step': 1.0,
                      })
        n_states = [environments.Rectangle.get_n_states(width, height) for width, height in
                    zip(par_env.widths, par_env.heights)]
    elif par.world_type in ['rectangle_chunk']:
        width, height = 2, 2
        par.seq_len = seq_len_mult * width * height
        par.torus = True
        # make chunks reasonably long, otherwise v easy to decode micro actions (5 good but too hard to learn. Try 4)
        # par.chunks = ['u,l', 'd,d,r', 'd,l,u', 'r,u,l', 'd,r', 'u,r']
        par.chunks = ['u,l,d,l,u', 'l,u,r,u,l', 'r,u,l,u,r', 'u,r,d,r,u',
                      'd,l,u,l,d', 'l,d,r,d,l', 'r,d,l,d,r', 'd,r,u,r,d']
        par.chunks = ['r,u,l,u', 'l,d,r,r', 'd,d,l,u', 'l,d,l,u',
                      'u,u,r,d', 'u,d,d,d', 'r,d,r,u', 'u,u,l,d']
        par.chunks = ['u,l,s,r', 'd,s,l,u', 'u,d,r,s', 's,d,l,r',
                      'd,d,r,u', 'l,l,d,r', 'u,u,l,d', 'r,r,u,l']
        name_dict = {'u': 'up',
                     'd': 'down',
                     'l': 'left',
                     'r': 'right',
                     's': 'stay still',
                     ',': ','}
        par.chunks = [''.join([name_dict[b] for b in x]) for x in par.chunks]
        par.n_chunk_actions = len(par.chunks) + 1
        par_env = Dd({'stay_still': True,
                      'bias_type': 'none',  # 'angle',
                      'direc_bias': 0.25,
                      'angle_bias_change': 0.4,
                      'widths': [width] * par.batch_size,
                      'heights': [height] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right', 'stay still'],
                      'velocities': [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]],
                      'dim_space': 2,
                      'correct_step': 1.0,
                      'n_chunk_actions': len(par.chunks) + 1
                      })
        n_states = [environments.RectangleChunk.get_n_states(width, height) for width, height in
                    zip(par_env.widths, par_env.heights)]
    elif par.world_type in ['rectangle_behave']:
        width, height = 2, 2
        par.seq_len = seq_len_mult * width * height
        par.torus = True
        par.behaviour_type = "random"

        par_env = Dd({'stay_still': False,
                      'bias_type': 'none',  # 'angle',
                      'direc_bias': 0.25,
                      'angle_bias_change': 0.4,
                      'widths': [width] * par.batch_size,
                      'heights': [height] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right', 'stay still'],
                      'velocities': [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]],
                      'dim_space': 2,
                      'correct_step': 1.0,
                      })
        n_states = [environments.RectangleBehave.get_n_states(width, height) for width, height in
                    zip(par_env.widths, par_env.heights)]
    elif par.world_type in ['Basu2021']:
        par.seq_len = 80
        par.torus = False
        # if Basu2019 -> width=10, height=1, pred_observation_is_position=True
        par.n_rewards = 2
        width, height = 10, 1

        par.repeat_path = True
        par.agent_initial_exploration = False
        par.start_at_first_reward = True
        par.only_train_on_rewards = True
        par.observation_is_position = True
        par.non_reward_are_delays_input = False
        par.non_reward_are_delays_target = True

        par_env = Dd({'stay_still': False,
                      'bias_type': 'angle',
                      'direc_bias': 0.25,
                      'angle_bias_change': 0.4,
                      'widths': [width] * par.batch_size,
                      'heights': [height] * par.batch_size,
                      'rels': ['down', 'up', 'left', 'right', 'stay still'],
                      'velocities': [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0], [0.0, 0.0]],
                      'dim_space': 2,
                      'correct_step': 1.0,
                      })
        n_states = [environments.RectangleRewards.get_n_states(width, height) for width, height in
                    zip(par_env.widths, par_env.heights)]

        if par.observation_is_position:
            par.o_size = np.max(n_states)
        if par.non_reward_are_delays_input or par.non_reward_are_delays_target:
            par.o_size = par.o_size + 1  # add extra observation for 'delay;
    elif par.world_type == 'loop':
        length = 4
        par.seq_len = seq_len_mult * length
        par.loops_before_rev = 1  # needs to be > 0
        # 'repeat'  'reverse' 'reverse_hierarchcial' 'skip_1' '1,-1,2,0' '1,2,0,-1' '2,1,0,-1' '2,1,0,-1,0,1',
        # '1,1,-1,1,1,0', '1,1,-1,1,1,0,1,0,-1,0,1'
        par.behaviour_type = "random"
        par_env = Dd({'stay_still': False,
                      'delay_max': 0,
                      'delay_min': 0,
                      'widths': [length] * par.batch_size,
                      'heights': [0] * par.batch_size,
                      'rels': ['forward_1', 'backward_1', 'forward_2', 'backward_2', 'stay still'],
                      'velocities': [1.0, -1.0, 2.0, -2.0, 0.0],
                      'dim_space': 1,
                      'repeat_vel_prob': 0.6,
                      })
        n_states = [environments.Loop.get_n_states_max(width) for width in par_env.widths]
    elif par.world_type == 'loop_chunk':
        length = 4
        par.seq_len = seq_len_mult * length
        par.chunks = ['1,0,1,1,1', '1,1,-1,1,1', '1,0,1,1,-1', '1,-1,1,-1,1', '1,1,-1,0,-1', '-1,1,-1,1,-1',
                      '-1,-1,-1,0,1', '-1,1,-1,-1,-1', '-1,-1,-1,0,-1']
        par.chunks = ['1,1,1,1', '1,0,1,1', '1,1,1,-1', '-1,1,0,1', '1,1,-1,-1', '0,-1,1,-1',
                      '-1,0,0,-1', '-1,-1,-1,0', '-1,-1,-1,-1']
        # par.chunks = ['1', '-1']
        par.n_chunk_actions = len(par.chunks) + 1
        par_env = Dd({'stay_still': False,
                      'delay_max': 0,
                      'delay_min': 0,
                      'widths': [length] * par.batch_size,
                      'heights': [0] * par.batch_size,
                      'rels': ['forward_1', 'backward_1', 'forward_2', 'backward_2', 'stay still'],
                      'velocities': [1.0, -1.0, 2.0, -2.0, 0.0],
                      'dim_space': 1,
                      'repeat_vel_prob': 0.4,
                      'n_chunk_actions': len(par.chunks) + 1
                      })
        n_states = [environments.LoopChunk.get_n_states_max(width) for width in par_env.widths]
    elif par.world_type == 'loop_delay':
        length = 4
        par.n_rewards = length
        par.o_size = par.o_size + 1
        # par.seq_len = seq_len_mult * length
        par.loops_before_rev = 1  # needs to be > 0
        par_env = Dd({
            'delay_max': 6,
            'delay_min': 3,
            'widths': [length] * par.batch_size,
            'heights': [0] * par.batch_size,
            'rels': ['forward_1', 'stay still'],
            'velocities': [1.0, 0.0],
            'dim_space': 1,
            'same_delays': False,
        })
        n_states = [environments.LoopDelay.get_n_states_max(width, par_env.delay_max) for width in par_env.widths]
        par.seq_len = seq_len_mult * int(length * (par_env.delay_max + par_env.delay_max + 2) / 2)
    elif par.world_type == 'loop_same_delay':
        length = 4
        par.n_rewards = length
        par.o_size = par.o_size + 1
        # par.seq_len = seq_len_mult * length
        par.loops_before_rev = 1  # needs to be > 0
        par_env = Dd({
            'delay_max': 6,
            'delay_min': 3,
            'widths': [length] * par.batch_size,
            'heights': [0] * par.batch_size,
            'rels': ['forward_1', 'stay still'],
            'velocities': [1.0, 0.0],
            'dim_space': 1,
            'same_delays': True,
        })
        n_states = [environments.LoopDelay.get_n_states_max(width, par_env.delay_max) for width in par_env.widths]
        par.seq_len = seq_len_mult * int(length * (par_env.delay_max + par_env.delay_max + 2) / 2)
    elif par.world_type == 'NBack':
        length = 4
        par.seq_len = seq_len_mult * length
        par_env = Dd({'widths': [length] * par.batch_size,
                      'heights': [0] * par.batch_size,
                      'rels': ['forward', 'stay still'],
                      'velocities': [1.0, 0.0],
                      'dim_space': 1,
                      })
        n_states = [environments.NBack.get_n_states(width) for width in par_env.widths]
    elif par.world_type == 'Panichello2021':
        length = 4
        par.n_rewards = length
        par.seq_len = length + 2
        par.o_size = par.o_size + length + 1  # 2 for cues, 1 for 'no observation'
        par_env = Dd({'widths': [length] * par.batch_size,
                      'heights': [0] * par.batch_size,
                      'rels': ['observation', 'stay_still'] + ['cue_' + str(x) for x in range(length)],
                      'velocities': [1.0, 0.0] + [x - length + 1.0 for x in range(length)],
                      # vel depends on how net chooses to structure slots... loop vs line etc...
                      'dim_space': 1,
                      })
        n_states = [environments.Panichello2021.get_n_states(width) for width in par_env.widths]
    elif par.world_type == 'Xie2022':
        length = 4
        par.n_rewards = length
        par.seq_len = length + length
        par.o_size = par.o_size + 1  # 1 for 'no observation' for second loop steps
        par_env = Dd({'widths': [length] * par.batch_size,
                      'heights': [0] * par.batch_size,
                      'rels': ['forward'],
                      'velocities': [1.0],
                      'dim_space': 1,
                      })
        n_states = [environments.Panichello2021.get_n_states(width) for width in par_env.widths]
    else:
        raise ValueError('incorrect world specified')

    # repeat widths and height
    par_env.widths = list(islice(cycle(par_env.widths), par.batch_size))
    par_env.heights = list(islice(cycle(par_env.heights), par.batch_size))

    par.max_states = np.max(n_states)
    par_env.n_actions = len(par_env.rels)
    par.n_actions = par_env.n_actions
    par.env = par_env
    return par


def get_scaling_parameters(index, par):
    # these scale with number of gradient updates
    l_r = (par.learning_rate_max - par.learning_rate_min) * (par.l_r_decay_rate ** (
            index / par.l_r_decay_steps)) + par.learning_rate_min
    l_r = np.maximum(l_r, par.learning_rate_min)

    scalings = Dd({'l_r': l_r,
                   'iteration': index,
                   })

    return scalings


def load_params_wrapper(save_dir, date, run):
    try:
        saved_path = save_dir + date + '/run' + str(run) + '/save'
        pars = load_params(saved_path)
    except FileNotFoundError:
        saved_path = save_dir + date + '/save/' + 'run' + str(run)
        pars = load_params(saved_path)
    return pars


def load_params(saved_path):
    try:
        pars = load_numpy_gz(saved_path + '/params.npy')
    except FileNotFoundError:
        pars = load_numpy_gz(saved_path + '/pars.npy')

    return Dd(pars.item())


def load_numpy_gz(file_name):
    try:
        return np.load(file_name, allow_pickle=True)
    except FileNotFoundError:
        f = gzip.GzipFile(file_name + '.gz', "r")
        return np.load(f, allow_pickle=True)


def get_params(save_dirs, date, run, not_this_dir=False, print_where=True):
    savedir1, params_1 = None, None
    if type(save_dirs) != list:
        save_dirs = [save_dirs]
    for save_dir in save_dirs:
        savedir1 = save_dir + date + '/run' + str(run)
        if savedir1 != not_this_dir:
            try:
                params_1 = load_params_wrapper(save_dir, date, run)
                if print_where:
                    print('params yes: ' + savedir1)
                break
            except FileNotFoundError:
                print('params not: ' + savedir1)
                pass
        else:
            print('params not: ' + savedir1)

    if print_where:
        print('')

    if params_1 is None:
        raise ValueError('NO PARAMS FOUND: ' + savedir1)
    else:
        saved_dir_1 = savedir1[:]

        return Dd(params_1), saved_dir_1


def compare_params(params_1, params_2, prename=''):
    messages = []
    for p1 in params_1:
        message = compare_param(params_1, params_2, p1, which=1, prename=prename)
        if type(message) == list:
            for m in message:
                messages.append(m)
        else:
            messages.append(message)

    for p2 in params_2:
        message = compare_param(params_1, params_2, p2, which=2, prename=prename)
        if type(message) == list:
            for m in message:
                messages.append(m)
        else:
            messages.append(message)

    messages = list(set(messages))

    return messages


def compare_param(params_1, params_2, param, which=1, prename=''):
    try:
        p1 = params_1[param]
        p2 = params_2[param]
    except KeyError:
        if which == 1:
            message = str(['missing param 2 : ', prename + '/' + param, params_1[param]])
        else:
            message = str(['missing param 1 : ', prename + '/' + param, params_2[param]])
        return message

    different = str(['different', prename + '/' + param, p1, p2])
    same = str(['same', prename + '/' + param, p1, p2])
    same_too_big = str(['same', prename + '/' + param, ' too big to show'])

    if not isinstance(p1, type(p2)):
        message = different
    elif type(p1) == dict or type(p2) == Dd:
        message = compare_params(params_1[param], params_2[param], prename=prename + '/' + param)
    elif type(p1) == list:
        if sorted(p1) == sorted(p2):
            message = same
        else:
            message = different
    elif type(p1) == np.ndarray:
        if np.array_equal(p1, p2):
            if np.max(np.shape(p1)) > 10:
                message = same_too_big
            else:
                message = same
        else:
            message = different
    else:
        if p1 == p2:
            message = same
        else:
            message = different

    return message


def find_model_with_params(save_dirs, keys, vals_desired):
    key = None
    for dirs in save_dirs:
        list_of_save_paths = [x[0] for x in os.walk(dirs) if
                              'save' in x[0][-20:] and 'run' in x[0] and 'iter' not in x[0]]
        list_of_save_paths.sort()
        for s_p in sorted(list_of_save_paths, reverse=True):
            print_bool = True
            try:
                pars = load_params(s_p)

                for key, val_desired in zip(keys, vals_desired):
                    if val_desired:
                        if pars[key[0]][key[1]] != val_desired:
                            print_bool = False
                    else:
                        # print(str([pars[key] for key in keys]) + ': ' + s_p)
                        pass
                if print_bool:
                    print(s_p)
                    print(str([pars[key[0]][key[1]] for key in keys]))

            except FileNotFoundError:
                # print('file not found: ' + s_p)
                pass
            except KeyError:
                pass
                # print(key, 'Key Error: ' + s_p)
    return


def convert_new_to_old_params(p1, p2):
    if ('data' in p1.keys()) != ('data' in p2.keys()):
        if 'data' in p1.keys():
            p1 = Dd({**p1.data, **p1.model, **p1.train, **p1.misc})
        if 'data' in p2.keys():
            p2 = Dd({**p2.data, **p2.model, **p2.train, **p2.misc})
    return p1, p2
