#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""
import itertools
import numpy as np
import data_utils as du
import model_utils as mu
import copy as cp
import parameters
import torch
import time
import importlib
import sklearn.linear_model as lm
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
import textwrap
from matplotlib.ticker import MaxNLocator


# import sklearn.ensemble as ens
# from sklearn import svm
# from sklearn.neural_network import MLPClassifier


# from line_profiler_pycharm import profile


def z_score(a):
    a_demean = a - np.mean(a)
    return a_demean / np.std(a)


def corr(a, b):
    return np.mean(z_score(a) * z_score(b))


def cosine_similarity(a, b):
    return np.sum(a * b) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))


def calc_mir(inp):
    # negative values not allowed
    inp = np.maximum(inp, 0.0)
    sum_ = np.sum(inp, axis=0)
    sum_[sum_ == 0.0] = 1.0
    return (np.mean(np.max(inp, axis=0) / sum_) - 1 / inp.shape[0]) / (1 - 1 / inp.shape[0])


# @profile
def compute_metrics(model, pars, batch_size=300, device='cpu', train_i=0, return_all=False, non_lin=False,
                    nonlin_override=False, decoding_analyses=True, parameters=parameters, load_para_script=None,
                    seq_len_mult=7, skip_slot_decoding=False):
    if load_para_script is not None:
        spec_parameters = importlib.util.spec_from_file_location("parameters", load_para_script + '/parameters.py')
        parameters = importlib.util.module_from_spec(spec_parameters)
        spec_parameters.loader.exec_module(parameters)
    # cells that are active at initial encoding, and have same activity at subsequent encodings.
    # (in theory could be observation cells...)
    start_time = time.time()
    metrics = {}

    if pars.data.world_type in ['Panichello2021', 'Xie2022']:
        # add specific analyses for these tasks...
        return metrics, time.time() - start_time

    """
    1. Using Embedding Weights
    """
    corrs, coses = [], []
    if 'observation' in pars.model.embedding_inputs:
        weight_in = model.embed_o.weight.detach().cpu().numpy()
        weight_in = weight_in[:, np.any(np.abs(weight_in) > np.abs(weight_in).max() / 20, axis=0)]
        for i, s in enumerate(weight_in):
            for j, s_ in enumerate(weight_in):
                if i >= j:
                    continue
                coses.append(cosine_similarity(s, s_))
                corrs.append(corr(s, s_))
        metrics['corr_input_weight'] = np.mean(corrs)
        metrics['cosine_input_weight'] = np.mean(coses)
        metrics['mir_input_weight'] = calc_mir(weight_in)

    """
    2. Using Readout Weights
    """
    corrs, coses = [], []
    for x in pars.model.to_predict:
        if x == 'target_o':
            try:
                weight_out = model.predict_t.weight.detach().cpu().numpy()
            except AttributeError:
                weight_out = model.predict_t[0].weight.detach().cpu().numpy()
        elif x == 'observation':
            try:
                weight_out = model.predict_o.weight.detach().cpu().numpy()
            except AttributeError:
                weight_out = model.predict_o[0].weight.detach().cpu().numpy()
        else:
            continue
        weight_out = weight_out[:, np.any(np.abs(weight_out) > np.abs(weight_out).max() / 20, axis=0)]
        for i, s in enumerate(weight_out):
            for j, s_ in enumerate(weight_out):
                if i >= j:
                    continue
                coses.append(cosine_similarity(s, s_))
                corrs.append(corr(s, s_))
        metrics['corr_output_weight_' + x] = np.mean(corrs)
        metrics['cosine_output_weight_' + x] = np.mean(coses)
        metrics['mir_output_weight_' + x] = calc_mir(weight_out)

    """
    3. Decoding analysis
    """
    params = parameters.default_params(batch_size=batch_size, h_size=pars.model.h_size)
    back_shifts = [x for x in range(params.data.max_states + 1) if x <= params.data.max_states]
    t_to_predict = {'observation': 0,
                    'target_o': 0,
                    'position': 0,
                    'reward': 0,
                    'phase_velocity': -1,
                    'phase_velocity_discrete_even': -1,
                    'phase_velocity_discrete_unique': -1,
                    'phase': 0,
                    'phase_discrete_even': 0,
                    'phase_discrete_unique': 0,
                    'velocity': -1,
                    'action': -1,
                    'goal_observation': 0,
                    'goal_position': 0,
                    }
    stop = params.data.seq_len + np.min([val for val in t_to_predict.values()])

    # clear variables from gpu
    torch.cuda.empty_cache()
    # Collect data
    train_dict = du.get_initial_data_dict(params.data, params.model.h_size)
    scalings = parameters.get_scaling_parameters(train_i, params.train)
    train_dict = du.data_step(train_dict, params.data)
    with torch.no_grad():
        inputs_torch = mu.inputs_2_torch(train_dict.inputs, scalings, device=device)
        variables, re_input = model(inputs_torch, device=device)

    # add extra for velocity phase...
    if params.data.world_type == 'loop':
        if params.data.behaviour_type in ['1,2,0,-1', '1,-1,2,0', '2,1,0,-1,0,1', '2,1,0,-1', '1,1,-1,1,1,0',
                                          '1,1,-1,1,1,0,1,0,-1,0,1']:
            vel_repeat = len([x for x in params.data.behaviour_type.split(',')])
            train_dict.walk_data.velocity_phase = [
                np.tile([x for x in range(vel_repeat)], int(params.data.seq_len / vel_repeat) + 1)[:params.data.seq_len]
                for _ in train_dict.walk_data.observation]
            t_to_predict['velocity_phase'] = 0
    elif params.data.world_type == 'rectangle_behave':
        if params.data.behaviour_type == 'up,left,down,down,right,right,up,up':
            train_dict.walk_data.velocity_phase = [
                np.tile([0, 1, 2, 3, 4, 5, 6, 7], int(params.data.seq_len / 8) + 1)[:params.data.seq_len] for _ in
                train_dict.walk_data.observation]
            t_to_predict['velocity_phase'] = 0

    if params.data.world_type in ['loop_delay', 'loop_same_delay']:
        num_phases = 4
        # get position on loop, rather than on big graph
        train_dict.walk_data.pos_0 = [(np.cumsum(r) - 1) % params.data.n_rewards for r in train_dict.walk_data.reward]
        train_dict.walk_data.pos_1 = [np.cumsum(r) % params.data.n_rewards for r in train_dict.walk_data.reward]
        train_dict.walk_data.pos_2 = [np.roll(r, 1) for r in train_dict.walk_data.pos_1]
        train_dict.walk_data.pos_3 = [(p0 + (ph >= 0.5)) % params.data.n_rewards for (p0, ph) in
                                      zip(train_dict.walk_data.pos_0, train_dict.walk_data.phase)]
        t_to_predict['pos_0'] = 0  # position if stay still during delay
        t_to_predict['pos_1'] = 0  # next position
        t_to_predict['pos_2'] = 0  # position if move immediately then stay still
        t_to_predict['pos_3'] = 0  # position if move slowly through delay

        # loop phase
        loop_phase = [np.round(np.cumsum(pv), 3) % params.data.n_rewards for pv in train_dict.walk_data.phase_velocity]
        train_dict.walk_data.loop_phase_discrete_even = phase_discrete([x / params.data.n_rewards for x in loop_phase],
                                                                       div_=2 * num_phases * params.data.n_rewards)
        train_dict.walk_data.loop_phase_discrete_unique = phase_discrete(
            [x / params.data.n_rewards for x in loop_phase])
        t_to_predict['loop_phase_discrete_even'] = 0
        t_to_predict['loop_phase_discrete_unique'] = 0

        # get 4-objects in order
        # e.g. loop: [2, 5, 1, 0]
        # observation_loop_phase_i : i-th loop observation at phase (e.g., 2nd loop observation (5) at phase 1.6-1.8)
        # observation_loop_steps_i : i-th loop observation at steps (e.g., 2nd loop observation (5) at phase 0.6-0.8)
        train_dict.walk_data['observation_loop_phase_discrete_even'] = []
        train_dict.walk_data['observation_loop_steps_discrete_even'] = []
        train_dict.walk_data['observation_loop_phase'] = []
        loop_objects = [x.states_mat[x.observation_locations] for x in train_dict.curric_env.envs]
        for obj_i in range(params.data.n_rewards):
            train_dict.walk_data['observation_loop_phase_discrete_even_' + str(obj_i)] = []
            train_dict.walk_data['observation_loop_steps_discrete_even_' + str(obj_i)] = []
            for env, l_p in enumerate(loop_phase):
                obj = loop_objects[env][obj_i]
                # how much loop_phase away from loop_phase 0 is observation
                ll_phase = ((l_p - obj_i) % params.data.n_rewards) / params.data.n_rewards
                l_phase_obj_discrete = phase_discrete(ll_phase, div_=2 * num_phases * params.data.n_rewards)
                # how many steps away from 0 is observation
                ll_steps = ((np.floor(l_p) - obj_i) % params.data.n_rewards) / params.data.n_rewards
                l_steps_obj_discrete = phase_discrete(ll_steps, div_=2 * params.data.n_rewards)
                # for decoding: object at what phase:
                # train_dict.walk_data['observation_loop_phase_discrete_even_' + str(obj_i)].append(
                #    l_phase_obj_discrete * params.data.o_size + obj)
                # train_dict.walk_data['observation_loop_steps_discrete_even_' + str(obj_i)].append(
                #    l_steps_obj_discrete * params.data.o_size + obj)

                # JW: FOR THIS TO WORK YOU NEED TO USE ALL OBJECTS IN EACH SLOT - i.e. DO SIMULTANEUSO MULTICLASS REGRESSION,
                # OR GO FROM (o1 in slotj, o2 in slot,...) to Neurons.

                # add one-hot encodings
                a = np.zeros((l_phase_obj_discrete.shape[0], num_phases * params.data.n_rewards * params.data.o_size))
                a[np.arange(l_phase_obj_discrete.shape[0]), l_phase_obj_discrete * params.data.o_size + obj] = 1.0
                b = np.zeros((l_phase_obj_discrete.shape[0], params.data.n_rewards * params.data.o_size))
                b[np.arange(l_phase_obj_discrete.shape[0]), l_steps_obj_discrete * params.data.o_size + obj] = 1.0
                c = np.zeros((ll_phase.shape[0], params.data.o_size))
                c[np.arange(ll_phase.shape[0]), obj] = ll_phase
                if obj_i == 0:
                    train_dict.walk_data['observation_loop_phase_discrete_even'].append(a)
                    train_dict.walk_data['observation_loop_steps_discrete_even'].append(b)
                    train_dict.walk_data['observation_loop_phase'].append(c)
                else:
                    train_dict.walk_data['observation_loop_phase_discrete_even'][env] += a
                    train_dict.walk_data['observation_loop_steps_discrete_even'][env] += b
                    train_dict.walk_data['observation_loop_phase'][env] += c
            # t_to_predict['observation_loop_phase_discrete_even+' + str(obj_i)] = 0
            # t_to_predict['observation_loop_steps_discrete_even_' + str(obj_i)] = 0
        t_to_predict['observation_loop_phase_discrete_even'] = 0
        t_to_predict['observation_loop_steps_discrete_even'] = 0
        t_to_predict['observation_loop_phase'] = 0
        shuffle = False
    else:
        shuffle = True

    average_firing = np.mean(variables.hidden.inf.detach().cpu().numpy(), axis=(0, 1))
    cell_to_keep = average_firing > average_firing.max() / 20
    metrics['n_cells_that_fire_above_one_twentieth_of_max'] = np.sum(cell_to_keep)

    # convert continuous to discrete
    train_dict.walk_data['phase_discrete_even'] = phase_discrete(train_dict.walk_data['phase'], div_=16)
    train_dict.walk_data['phase_discrete_unique'] = phase_discrete(train_dict.walk_data['phase'])
    train_dict.walk_data['phase_velocity_discrete_even'] = phase_discrete(train_dict.walk_data['phase_velocity'],
                                                                          div_=16)
    train_dict.walk_data['phase_velocity_discrete_unique'] = phase_discrete(train_dict.walk_data['phase_velocity'])

    # get `fully explored' locations, i.e. all slots potentially filled
    fully_explored = [np.logical_and(exp == 0, np.cumsum(exp) == max(np.cumsum(exp))) for exp, env in
                      zip(train_dict.inputs.exploration.T, train_dict.curric_env.envs)]
    starts = [env.n_states for env in train_dict.curric_env.envs]
    # remove 'start' and 'end' states
    fully_explored = [x[start:stop] for x, start in zip(fully_explored, starts)]

    # get hidden activations
    hidden = variables.hidden.inf.detach().cpu().numpy()
    # remove hidden not explored
    hidden = [x[start:stop][fe] for x, fe, start in zip(hidden.transpose((1, 0, 2)), fully_explored, starts)]

    n_train = int(batch_size * 0.8)
    x_train = np.concatenate(hidden[:n_train], axis=0)
    x_test = np.concatenate(hidden[n_train:], axis=0)

    do_nonlin = (non_lin and train_i > 0 and train_i % (
            params.misc.sum_int * params.misc.mult_sum_metrics * 5) == 0) or nonlin_override
    factors_n_back, factors_loop_slots, factors_spatial_slots = None, None, None
    n_back_coefs, loop_coefs, spatial_coefs = None, None, None
    hidden_dim = 200
    logistic_reg_max_its = 150
    for pred_type in t_to_predict.keys():
        # if variable always constant then don't bother... (ignore 1st one as that's often special
        if len(set([item for sublist in
                    [x[start:, ...].flatten() for x, start in zip(train_dict.walk_data[pred_type], starts)] for item in
                    sublist])) <= 1:
            continue
        n = t_to_predict[pred_type]
        # get data, making sure you shift and only take fully explored
        y = [x[start - n:stop - n][fe] for x, fe, start in zip(train_dict.walk_data[pred_type], fully_explored, starts)]
        y_train = np.concatenate(y[:n_train], axis=0)
        y_test = np.concatenate(y[n_train:], axis=0)

        if len(set(y_train.flatten())) <= 1:
            breakpoint()

        if decoding_analyses:
            if pred_type in ['observation_loop_phase_discrete_even', 'observation_loop_steps_discrete_even']:
                func = MultiLogisticRegression(hidden_dim, y_train.shape[1], params.model.h_size)
            else:
                func = lm.LinearRegression() if 'float' in y_train.dtype.name else \
                    lm.LogisticRegression(random_state=0, class_weight='balanced', max_iter=logistic_reg_max_its)
            clf = func.fit(x_train, y_train)
            metrics['decode_' + pred_type + '_around current_' + str(n) + '_train'] = clf.score(x_train, y_train)
            metrics['decode_' + pred_type + '_around current_' + str(n) + '_test'] = clf.score(x_test, y_test)

            if 'chunk' in params.data.world_type and pred_type in ['action']:
                # compute decoder accuracy but for 0th, 1th, 2th etc after chunk input
                # get micro action 'locations'
                y_pred_ = clf.predict(x_test)
                correct_predictions_ = np.array([1 if y_pred_[i] == y_test[i] else 0 for i in range(len(y_pred_))])

                a_ = []
                for ii, aa in enumerate(train_dict.walk_data['chunk_action']):
                    kk = 0
                    b_ = []
                    for jj, bb in enumerate(aa):
                        if bb != 0:
                            kk = 0
                        else:
                            kk += 1
                        b_.append(kk)
                    a_.append(np.array(b_))

                n_ = -1
                y_ = [x[start - n_:stop - n_][fe] for x, fe, start in zip(a_, fully_explored, starts)]
                n_after_chunk_ = np.concatenate(y_[n_train:], axis=0)

                for n_ in np.unique(n_after_chunk_):
                    metrics['decode_' + pred_type + '_around_chunk_' + str(n_) + '_test'] = np.mean(
                        correct_predictions_[np.where(n_after_chunk_ == n_)[0]])

            if do_nonlin and 'float' not in y_train.dtype.name:
                # do the same for non-linear classifier:
                func = SimpleKerasModel(hidden_dim, len(np.unique(y_train)), params.model.h_size)
                clf = func.fit(x_train, y_train)
                metrics['decode_' + pred_type + '_around current_' + str(n) + '_nonlin_train'] = clf.score(x_train,
                                                                                                           y_train)
                metrics['decode_' + pred_type + '_around current_' + str(n) + '_nonlin_test'] = clf.score(x_test,
                                                                                                          y_test)
            if shuffle:
                # compare to shuffle
                y_train = np.concatenate([np.random.permutation(x) for x in y[:n_train]], axis=0)
                y_test = np.concatenate([np.random.permutation(x) for x in y[n_train:]], axis=0)

                if pred_type in ['observation_loop_phase_discrete_even', 'observation_loop_steps_discrete_even']:
                    func = MultiLogisticRegression(hidden_dim, y_train.shape[1], params.model.h_size)
                else:
                    func = lm.LinearRegression() if 'float' in y_train.dtype.name else \
                        lm.LogisticRegression(random_state=0, class_weight='balanced', max_iter=logistic_reg_max_its)
                clf = func.fit(x_train, y_train)
                metrics['decode_' + pred_type + '_around current_' + str(n) + '_shuffle_train'] = clf.score(x_train,
                                                                                                            y_train)
                metrics['decode_' + pred_type + '_around current_' + str(n) + '_shuffle_test'] = clf.score(x_test,
                                                                                                           y_test)

                if do_nonlin and 'float' not in y_train.dtype.name:
                    # do the same for non-linear classifier:
                    func = SimpleKerasModel(hidden_dim, len(np.unique(y_train)), params.model.h_size)
                    clf = func.fit(x_train, y_train)
                    metrics['decode_' + pred_type + '_around current_' + str(n) + '_nonlin_shuffle_train'] = clf.score(
                        x_train, y_train)
                    metrics['decode_' + pred_type + '_around current_' + str(n) + '_nonlin_shuffle_test'] = clf.score(
                        x_test, y_test)

        if params.data.world_type in ['loop_delay'] and (pred_type in ['observation'] and not skip_slot_decoding):
            # work out which phase velocity should be in which phase slot...
            name = '_loop_delay_slot_'

            for pred_type_ in ['phase_velocity', 'phase_velocity_discrete_even', 'phase_velocity_discrete_unique']:
                loop_coefs = []
                loop_train_all, loop_test_all = [], []
                loop_nlin_train_all, loop_nlin_test_all = [], []

                state_vel = [[pv[max(np.where(np.roll(pos, 1) == ii)[0])] for ii in
                              np.arange(params.data.n_rewards)] for pos, pv
                             in zip(train_dict.walk_data.pos_0, train_dict.walk_data[pred_type_])]

                for slot_n in np.arange(params.data.n_rewards):
                    y = [x[start:stop][fe] for x, fe, start in zip(train_dict.walk_data.pos_0, fully_explored, starts)]

                    slot_n_pos = [pos_in_loop_slot(pos, slot_n, env.width) for pos, env in
                                  zip(y, train_dict.curric_env.envs)]
                    slot_n_obs = [np.array(env_)[pos] for pos, env_ in zip(slot_n_pos, state_vel)]

                    y_train = np.concatenate(slot_n_obs[:n_train], axis=0)
                    y_test = np.concatenate(slot_n_obs[n_train:], axis=0)

                    if not decoding_analyses:
                        continue

                    func = lm.LinearRegression() if 'float' in y_train.dtype.name else \
                        lm.LogisticRegression(random_state=0, class_weight='balanced', max_iter=logistic_reg_max_its)

                    clf = func.fit(x_train, y_train)
                    metrics['decode_' + pred_type_ + name + str(slot_n) + '_train'] = clf.score(x_train, y_train)
                    metrics['decode_' + pred_type_ + name + str(slot_n) + '_test'] = clf.score(x_test, y_test)

                    if 0 < slot_n:  # don't consider at immediate slot
                        loop_coefs.append(clf.coef_)
                        loop_train_all.append(metrics['decode_' + pred_type_ + name + str(slot_n) + '_train'])
                        loop_test_all.append(metrics['decode_' + pred_type_ + name + str(slot_n) + '_test'])

                    if do_nonlin:
                        # do the same for non-linear classifier: Random forest.
                        # func = svm.LinearSVC(dual="auto")
                        # func = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,), random_state=1)
                        func = SimpleKerasModel(hidden_dim, len(np.unique(y_train)), params.model.h_size)
                        # func = ens.RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=4)
                        clf = func.fit(x_train, y_train)
                        metrics['decode_' + pred_type_ + name + str(slot_n) + '_nonlin_train'] = clf.score(x_train,
                                                                                                          y_train)
                        metrics['decode_' + pred_type_ + name + str(slot_n) + '_nonlin_test'] = clf.score(x_test, y_test)
                        if 0 < slot_n:  # don't consider at immediate slot
                            loop_nlin_train_all.append(
                                metrics['decode_' + pred_type_ + name + str(slot_n) + '_nonlin_train'])
                            loop_nlin_test_all.append(
                                metrics['decode_' + pred_type_ + name + str(slot_n) + '_nonlin_test'])

                # loop slots mean
                metrics['decode_' + pred_type_ + name + 'mean_train'] = np.mean(loop_train_all)
                metrics['decode_' + pred_type_ + name + 'mean_test'] = np.mean(loop_test_all)
                if do_nonlin:
                    # do the same for non-linear classifier: Random forest.
                    metrics['decode_' + pred_type_ + name + 'nonlin_mean_train'] = np.mean(loop_nlin_train_all)
                    metrics['decode_' + pred_type_ + name + 'nonlin_mean_test'] = np.mean(loop_nlin_test_all)

        if params.data.world_type not in ['loop', 'rectangle', 'rectangle_behave', 'line', 'NBack', 'loop_chunk',
                                          'rectangle_chunk'] or pred_type not in ['observation'] or skip_slot_decoding:
            continue

        # THIS IS DECODING ON N_BACK OBSERVATIONS ETC
        factors_n_back = []
        n_back_coefs = []
        n_back_train_all, n_back_test_all = [], []
        n_back_nlin_train_all, n_back_nlin_test_all = [], []
        name = '_n_back_'
        for n in back_shifts:
            # JW : THIS FAILS WITH NBACK AND LOOP-REPEAT. IT IS BECAUSE START IS TOO SMALL FOR LARGE n
            y = [x[start - n:stop - n][fe] for x, fe, start in
                 zip(train_dict.walk_data[pred_type], fully_explored, starts)]
            y_train = np.concatenate(y[:n_train], axis=0)
            y_test = np.concatenate(y[n_train:], axis=0)
            if n < params.data.max_states:
                # only include up to max states back
                factors_n_back.append(np.concatenate(y, axis=0))

            if not decoding_analyses:
                continue

            func = lm.LinearRegression() if pred_type == 'velocity' else lm.LogisticRegression(
                random_state=0, class_weight='balanced', max_iter=logistic_reg_max_its)
            clf = func.fit(x_train, y_train)
            metrics['decode_' + pred_type + name + str(n) + '_train'] = clf.score(x_train, y_train)
            metrics['decode_' + pred_type + name + str(n) + '_test'] = clf.score(x_test, y_test)
            if 0 < n < params.data.max_states:  # ignore first and beyond max_states for mean
                n_back_coefs.append(clf.coef_)
                n_back_train_all.append(metrics['decode_' + pred_type + name + str(n) + '_train'])
                n_back_test_all.append(metrics['decode_' + pred_type + name + str(n) + '_test'])

            if do_nonlin:
                # do the same for non-linear classifier: Random forest.
                # func = svm.LinearSVC(dual="auto")
                # raise ValueError('STOP!')
                func = SimpleKerasModel(hidden_dim, len(np.unique(y_train)), params.model.h_size)
                # func = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,), random_state=1)
                # func = ens.RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=4)
                clf = func.fit(x_train, y_train)
                metrics['decode_' + pred_type + name + str(n) + '_nonlin_train'] = clf.score(x_train, y_train)
                metrics['decode_' + pred_type + name + str(n) + '_nonlin_test'] = clf.score(x_test, y_test)
                if 0 < n < params.data.max_states:  # ignore first and beyond max_states for mean
                    n_back_nlin_train_all.append(metrics['decode_' + pred_type + name + str(n) + '_nonlin_train'])
                    n_back_nlin_test_all.append(metrics['decode_' + pred_type + name + str(n) + '_nonlin_test'])

        # n back mean (only average up to max states back)
        metrics['decode_' + pred_type + name + 'mean_train'] = np.mean(n_back_train_all)
        metrics['decode_' + pred_type + name + 'mean_test'] = np.mean(n_back_test_all)
        if do_nonlin:
            # do the same for non-linear classifier: Random forest.
            metrics['decode_' + pred_type + name + 'nonlin_mean_train'] = np.mean(n_back_nlin_train_all)
            metrics['decode_' + pred_type + name + 'nonlin_mean_test'] = np.mean(n_back_nlin_test_all)

        # THIS IS DECODING FOR SLOTS THAT ARE ORGANISED ON A STRUCTURE - LOOP or 2D SPACE (Torus)
        if params.data.world_type in ['loop', 'line', 'loop_chunk']:
            # 1D Structure
            # slot(n) at time_step t will be filled with observation at position: mod(position at timestep t - n, n_pos)
            # s(n)_t = o[mod(p_t - n , n_pos)]
            # good way to distinguish this way fo thinking to whether we can decode 1-step back, 2 steps back and so on.
            # the above way of thinking has 'abstract' position embedded in how slots move :)
            factors_loop_slots = []
            loop_coefs = []
            loop_train_all, loop_test_all = [], []
            loop_nlin_train_all, loop_nlin_test_all = [], []
            name = '_loop_slot_'
            for slot_n in np.arange(params.data.max_states):
                y = [x[start:stop][fe] for x, fe, start in zip(train_dict.walk_data.position, fully_explored, starts)]
                slot_n_pos = [pos_in_loop_slot(pos, slot_n, env.width) for pos, env in
                              zip(y, train_dict.curric_env.envs)]
                slot_n_obs = [env_.states_mat[pos] for pos, env_ in zip(slot_n_pos, train_dict.curric_env.envs)]

                y_train = np.concatenate(slot_n_obs[:n_train], axis=0)
                y_test = np.concatenate(slot_n_obs[n_train:], axis=0)

                factors_loop_slots.append(np.concatenate(slot_n_obs, axis=0))
                if not decoding_analyses:
                    continue

                func = lm.LogisticRegression(random_state=0, class_weight='balanced', max_iter=logistic_reg_max_its)
                clf = func.fit(x_train, y_train)
                metrics['decode_' + pred_type + name + str(slot_n) + '_train'] = clf.score(x_train, y_train)
                metrics['decode_' + pred_type + name + str(slot_n) + '_test'] = clf.score(x_test, y_test)

                if 0 < slot_n:  # don't consider at immediate slot
                    loop_coefs.append(clf.coef_)
                    loop_train_all.append(metrics['decode_' + pred_type + name + str(slot_n) + '_train'])
                    loop_test_all.append(metrics['decode_' + pred_type + name + str(slot_n) + '_test'])

                if do_nonlin:
                    # do the same for non-linear classifier: Random forest.
                    # func = svm.LinearSVC(dual="auto")
                    # func = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,), random_state=1)
                    func = SimpleKerasModel(hidden_dim, len(np.unique(y_train)), params.model.h_size)
                    # func = ens.RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=4)
                    clf = func.fit(x_train, y_train)
                    metrics['decode_' + pred_type + name + str(slot_n) + '_nonlin_train'] = clf.score(x_train, y_train)
                    metrics['decode_' + pred_type + name + str(slot_n) + '_nonlin_test'] = clf.score(x_test, y_test)
                    if 0 < slot_n:  # don't consider at immediate slot
                        loop_nlin_train_all.append(
                            metrics['decode_' + pred_type + name + str(slot_n) + '_nonlin_train'])
                        loop_nlin_test_all.append(metrics['decode_' + pred_type + name + str(slot_n) + '_nonlin_test'])

            # loop slots mean
            metrics['decode_' + pred_type + name + 'mean_train'] = np.mean(loop_train_all)
            metrics['decode_' + pred_type + name + 'mean_test'] = np.mean(loop_test_all)
            if do_nonlin:
                # do the same for non-linear classifier: Random forest.
                metrics['decode_' + pred_type + name + 'nonlin_mean_train'] = np.mean(loop_nlin_train_all)
                metrics['decode_' + pred_type + name + 'nonlin_mean_test'] = np.mean(loop_nlin_test_all)

        elif params.data.world_type in ['rectangle', 'rectangle_behave', 'rectangle_chunk']:

            # 2D Structure.
            # Here slots will be like [[s00, s01, s02] ; [s10, s11, s12] ; [s20, s21, s22]]
            # If we take a left/right/up/down then contents get shifted to left/right/up/down
            # So should be able to decode
            factors_spatial_slots = []
            spatial_coefs = []
            spatial_train_all, spatial_test_all = [], []
            spatial_nlin_train_all, spatial_nlin_test_all = [], []
            name = '_spatial_slot_'
            for slot_n in np.arange(params.data.max_states):
                y = [x[start:stop][fe] for x, fe, start in zip(train_dict.walk_data.position, fully_explored, starts)]
                slot_n_pos = [pos_in_spatial_slot(pos, slot_n, env.width, env.height) for pos, env in
                              zip(y, train_dict.curric_env.envs)]
                slot_n_obs = [env_.states_mat[pos] for pos, env_ in zip(slot_n_pos, train_dict.curric_env.envs)]

                y_train = np.concatenate(slot_n_obs[:n_train], axis=0)
                y_test = np.concatenate(slot_n_obs[n_train:], axis=0)

                factors_spatial_slots.append(np.concatenate(slot_n_obs, axis=0))
                if not decoding_analyses:
                    continue

                func = lm.LogisticRegression(random_state=0, class_weight='balanced', max_iter=logistic_reg_max_its)
                clf = func.fit(x_train, y_train)
                metrics['decode_' + pred_type + name + str(slot_n) + '_train'] = clf.score(x_train, y_train)
                metrics['decode_' + pred_type + name + str(slot_n) + '_test'] = clf.score(x_test, y_test)

                if 0 < slot_n:  # don't consider at immediate slot
                    spatial_coefs.append(clf.coef_)
                    spatial_train_all.append(metrics['decode_' + pred_type + name + str(slot_n) + '_train'])
                    spatial_test_all.append(metrics['decode_' + pred_type + name + str(slot_n) + '_test'])

                if do_nonlin:
                    # do the same for non-linear classifier: Random forest.
                    # func = svm.LinearSVC(dual="auto")
                    # func = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200,), random_state=1)
                    func = SimpleKerasModel(hidden_dim, len(np.unique(y_train)), params.model.h_size)
                    # func = ens.RandomForestClassifier(random_state=0, class_weight='balanced', max_depth=4)
                    clf = func.fit(x_train, y_train)
                    metrics['decode_' + pred_type + name + str(slot_n) + '_nonlin_train'] = clf.score(x_train, y_train)
                    metrics['decode_' + pred_type + name + str(slot_n) + '_nonlin_test'] = clf.score(x_test, y_test)
                    if 0 < slot_n:  # don't consider at immediate slot
                        spatial_nlin_train_all.append(
                            metrics['decode_' + pred_type + name + str(slot_n) + '_nonlin_train'])
                        spatial_nlin_test_all.append(
                            metrics['decode_' + pred_type + name + str(slot_n) + '_nonlin_test'])

            # spatial slots mean
            metrics['decode_' + pred_type + name + 'mean_train'] = np.mean(spatial_train_all)
            metrics['decode_' + pred_type + name + 'mean_test'] = np.mean(spatial_test_all)
            if do_nonlin:
                # do the same for non-linear classifier: Random forest.
                metrics['decode_' + pred_type + name + 'nonlin_mean_train'] = np.mean(spatial_nlin_train_all)
                metrics['decode_' + pred_type + name + 'nonlin_mean_test'] = np.mean(spatial_nlin_test_all)

    """
    4. Mutual information
    """
    m_current, m_n_back, m_loop_slots, m_spatial_slots, m_n_back_noslot1, m_loop_slots_noslot1, \
        m_spatial_slots_noslot1, proj_onto_pca, slot_coords = None, None, None, None, None, None, None, None, None
    if pars.data.world_type in ['loop', 'NBack', 'rectangle', 'rectangle_behave', 'loop_chunk', 'rectangle_chunk']:
        factors_current = []
        hidden_ = np.concatenate([x[:, cell_to_keep] for x in hidden], axis=0).T
        # current:
        for pred_type in t_to_predict.keys():
            # if variable always constant then don't bother... (ignore 1st one as that's often special
            if train_dict.walk_data[pred_type][0] is None:
                continue
            if len(set([item for sublist in
                        [x[start:, ...].flatten() for x, start in zip(train_dict.walk_data[pred_type], starts)] for item
                        in sublist])) <= 1:
                continue
            if len(train_dict.walk_data[pred_type][0].shape) > 1:
                continue
            n = 0
            y = [x[start - n:stop - n][fe] for x, fe, start in
                 zip(train_dict.walk_data[pred_type], fully_explored, starts)]
            factors_current.append(np.concatenate(y, axis=0))

        # current
        factors_current = np.stack(factors_current, axis=0)
        metrics['mir_current'], m_current = compute_mir(factors_current, hidden_)
        # n_back
        if factors_n_back is not None:
            factors_n_back = np.stack(factors_n_back, axis=0)
            metrics['mir_n_back'], m_n_back = compute_mir(factors_n_back, hidden_)
            # remove 'slot 1 cells'
            non_slot1_cells = np.argmax(m_n_back, axis=1) != 0
            if np.sum(non_slot1_cells) > 0:
                metrics['mir_n_back_noslot1'], m_n_back_noslot1 = compute_mir(
                    factors_n_back[1:], hidden_[non_slot1_cells])
            else:
                metrics['mir_n_back_noslot1'] = 0.0
        else:
            m_n_back, m_n_back_noslot1 = None, None
        # loop slots
        if factors_loop_slots is not None:
            factors_loop_slots = np.stack(factors_loop_slots, axis=0)
            metrics['mir_loop_slots'], m_loop_slots = compute_mir(factors_loop_slots, hidden_)
            # remove 'slot 1 cells'
            non_slot1_cells = np.argmax(m_loop_slots, axis=1) != 0
            if np.sum(non_slot1_cells) > 0:
                metrics['mir_loop_slots_noslot1'], m_loop_slots_noslot1 = compute_mir(
                    factors_loop_slots[1:], hidden_[non_slot1_cells])
            else:
                metrics['mir_loop_slots_noslot1'] = 0.0
        else:
            m_loop_slots, m_loop_slots_noslot1 = None, None
        # spatial slots
        if factors_spatial_slots is not None:
            factors_spatial_slots = np.stack(factors_spatial_slots, axis=0)
            metrics['mir_spatial_slots'], m_spatial_slots = compute_mir(factors_spatial_slots, hidden_)
            # remove 'slot 1 cells'
            non_slot1_cells = np.argmax(m_spatial_slots, axis=1) != 0
            if np.sum(non_slot1_cells) > 0:
                metrics['mir_spatial_slots_noslot1'], m_spatial_slots_noslot1 = compute_mir(
                    factors_spatial_slots[1:], hidden_[non_slot1_cells])
            else:
                metrics['mir_spatial_slots_noslot1'] = 0.0
        else:
            m_spatial_slots, m_spatial_slots_noslot1 = None, None

        # spatial organisation of slots
        proj_onto_pca, slot_coords = None, None
        if 'weight_spatial_l1' in pars.train.which_costs or 'weight_spatial_l2' in pars.train.which_costs:
            n_idx = np.arange(pars.model.h_size)
            dim = np.ceil(pars.model.h_size ** 0.5)
            # normalise between 0 and 1
            dx = (n_idx[None, ...] % dim - n_idx[..., None] % dim) / (dim - 1)
            dy = (n_idx[None, ...] // dim - n_idx[..., None] // dim) / (dim - 1)
            distances = (dx ** 2 + dy ** 2) ** 0.5
            inner = distances < distances.max() / 5
            # go to cells to keep
            inner = inner[:, cell_to_keep]
            inner = inner[cell_to_keep, :]
            # are rnn weights bigger for nearby weights
            rnn_weight = model.transition.weight.detach().cpu().numpy()
            rnn_weight = rnn_weight[:, cell_to_keep]
            rnn_weight = rnn_weight[cell_to_keep, :]
            metrics['far_near_weight_ratio'] = np.mean(np.abs(rnn_weight[~inner])) / np.mean(np.abs(rnn_weight[inner]))
            # how similar pairs of cells are in terms of slot preference
            a = np.mean((m_n_back[None, ...] - m_n_back[:, None, ...]) ** 2, axis=2)
            # Are similarly slot pref cells closeby?
            metrics['spatial_slot_nback'] = np.mean(a * inner) / np.mean(a)
            a = np.abs(np.argmax(m_n_back, axis=1)[None, ...] - np.argmax(m_n_back, axis=1)[:, None, ...])
            metrics['spatial_slot_nback_argmax'] = np.mean(a * inner) / np.mean(a)
            # do slot preferences lie in order on a line?
            # compute pca of (x,y) coords for each 'slot', then take 1st dimension -- this is line
            # (probs only use active neurons...)
            # then see how ordered along line (number positions away from true position?)
            slot_pref = np.argmax(m_n_back, axis=1)
            metrics['num_slots_nback_preferred'] = len(np.unique(slot_pref))
            x_ = (n_idx[cell_to_keep] % dim) / (dim - 1)
            y_ = (n_idx[cell_to_keep] // dim) / (dim - 1)
            slot_coords = []
            for sp in range(m_n_back.shape[1]):
                idx = (slot_pref == sp)
                if sum(idx) == 0:
                    slot_coords.append([np.random.rand(), np.random.rand()])
                else:
                    slot_coords.append([np.mean(x_[idx]), np.mean(y_[idx])])
            slot_coords = np.asarray(slot_coords)
            pca = PCA(n_components=1)
            pca.fit(slot_coords)
            proj_onto_pca = pca.transform(slot_coords).flatten()
            order = np.argsort(proj_onto_pca)
            # x, y = np.stack([pca.mean_ - 0.5 * pca_0, pca.mean_ + 0.5 * pca_0]).T
            metrics['spatial_slot_nback_ordered_spearmanrank'] = np.abs(
                spearmanr(a=np.arange(m_n_back.shape[1]), b=order)[0])
            metrics['spatial_slot_nback_ordered_spearmanrank_shuffle'] = np.abs(
                spearmanr(a=np.random.permutation(np.arange(m_n_back.shape[1])), b=order)[0])
            metrics['spatial_slot_nback_ordered_corrcoef'] = np.abs(
                np.corrcoef(np.arange(m_n_back.shape[1]), proj_onto_pca)[1, 0])
            metrics['spatial_slot_nback_ordered_corrcoef_shuffle'] = np.abs(
                np.corrcoef(np.random.permutation(np.arange(m_n_back.shape[1])), proj_onto_pca)[1, 0])

        """
        5. SLOT ORTHOGONALITY
        """
        if decoding_analyses:
            order = ['NBack', 'Loop', 'Spatial']
            for name, coefs in zip(order, [n_back_coefs, loop_coefs, spatial_coefs]):
                if coefs is None:
                    continue
                for i in range(2):
                    if i == 1:
                        o = ortho_group.rvs(coefs[0].shape[1])
                        coefs = [np.matmul(x, o) for x in coefs]
                        extra = '_random_rotation'
                    else:
                        extra = ''
                    # neurons can only be positive, so we can just look at positive weights
                    coefs_processed = [np.mean(np.abs(x), axis=0) for x in coefs]
                    # compute dot product between slots
                    combinations = itertools.combinations(coefs_processed, 2)
                    cos_thetas = []
                    for cb in combinations:
                        cos_thetas.append(
                            np.sum(cb[0] * cb[1]) / ((np.sum(cb[0] * cb[0]) * np.sum(cb[1] * cb[1])) ** 0.5))
                    metrics['slot_cosine_mean_' + name + extra] = np.mean(cos_thetas)

                metrics['slot_cosine_mean_' + name + '_ratio'] = metrics['slot_cosine_mean_' + name] / (
                        metrics['slot_cosine_mean_' + name] +
                        metrics['slot_cosine_mean_' + name + '_random_rotation'] + 1e-8)

    """
    6. ARE REPS THE SAME ON RETURN TO LOCATION?? 
    Only do this when all locations have been visited.
    """
    if pars.data.world_type in ['loop', 'NBack', 'rectangle', 'rectangle_behave']:
        positions = [x[start:stop][fe] for x, fe, start in zip(train_dict.walk_data.position, fully_explored, starts)]
        diffs_all = []
        diffs_shuffle_all = []
        for pos, hid in zip(positions, hidden):
            diffs = []
            diffs_shuffle = []
            for p in np.unique(pos):
                hids = hid[np.where(pos == p)[0]]
                for i, hid_i in enumerate(hids):
                    diffs_shuffle.append(np.mean((hid_i - hid[np.random.randint(hid.shape[0])]) ** 2))
                    for j, hid_j in enumerate(hids):
                        if i < j:
                            diffs.append(np.mean((hid_i - hid_j) ** 2))
            if len(diffs) > 0:
                diffs_all.append(np.mean(diffs))
                diffs_shuffle_all.append(np.mean(diffs_shuffle))

        metrics['rep_diff_on_return'] = np.mean(diffs_all)
        metrics['rep_diff_on_return_shuffle'] = np.mean(diffs_shuffle_all)
        metrics['rep_diff_on_return_ratio'] = metrics['rep_diff_on_return'] / (
                metrics['rep_diff_on_return'] + metrics['rep_diff_on_return_shuffle'] + 1e-10)

        # clear variables from gpu
        del hidden, variables, train_dict, inputs_torch
        torch.cuda.empty_cache()

    """
    7. SLOT ALGEBRA
    """
    diffs, diffs_shuffle, diffs_1_4 = None, None, None
    if pars.data.world_type in ['loop', 'NBack', 'rectangle', 'rectangle_behave']:
        """
        A) Different Scenes, Different Sequences
        """
        diffs, diffs_shuffle, diffs_1_4 = [], [], []
        params = parameters.default_params(batch_size=100, h_size=pars.model.h_size)
        scalings = parameters.get_scaling_parameters(train_i, params.train)
        for i in range(8):
            train_dict = du.get_initial_data_dict(params.data, params.model.h_size)
            train_dict = du.data_step(train_dict, params.data, algebra='yes')
            with torch.no_grad():
                inputs_torch = mu.inputs_2_torch(train_dict.inputs, scalings, device=device)
                variables, re_input = model(inputs_torch, device=device)
                hidden = mu.torch2numpy(variables.hidden.inf)
            diffs_, diffs_shuffle_, diffs_1_4_ = slot_algebra_analysis(train_dict, hidden, params)
            diffs.extend(diffs_)
            diffs_shuffle.extend(diffs_shuffle_)
            diffs_1_4.extend(diffs_1_4_)
        metrics['slot_algebra_scene'], metrics['slot_algebra_scene_shuffle'], metrics['slot_algebra_scene_1_4'] = \
            np.mean(diffs), np.mean(diffs_shuffle), np.mean(diffs_1_4)
        metrics['slot_algebra_scene_shuffle_ratio'] = np.mean([(x + 1e-12) / (x + y + 2e-12) for x, y in
                                                               zip(diffs, diffs_shuffle)])
        metrics['slot_algebra_scene_1_4_ratio'] = np.mean([(x + 1e-12) / (x + y + 2e-12) for x, y in
                                                           zip(diffs, diffs_1_4)])
        # clear variables from gpu  torch.cuda.empty_cache()
        del hidden, variables, train_dict, inputs_torch
        torch.cuda.empty_cache()

        """
        B) Different Scenes, Same Sequences
        """
        diffs, diffs_shuffle, diffs_1_4 = [], [], []
        for i in range(8):
            train_dict = du.get_initial_data_dict(params.data, params.model.h_size)
            train_dict = du.data_step(train_dict, params.data, algebra='seq')
            with torch.no_grad():
                inputs_torch = mu.inputs_2_torch(train_dict.inputs, scalings, device=device)
                variables, re_input = model(inputs_torch, device=device)
                hidden = mu.torch2numpy(variables.hidden.inf)
            diffs_, diffs_shuffle_, diffs_1_4_ = slot_algebra_analysis(train_dict, hidden, params, same_index=True)
            diffs.extend(diffs_)
            diffs_shuffle.extend(diffs_shuffle_)
            diffs_1_4.extend(diffs_1_4_)

        metrics['slot_algebra_seq'], metrics['slot_algebra_seq_shuffle'], metrics['slot_algebra_seq_1_4'] = \
            np.mean(diffs), np.mean(diffs_shuffle), np.mean(diffs_1_4)
        metrics['slot_algebra_seq_shuffle_ratio'] = np.mean([(x + 1e-12) / (x + y + 2e-12) for x, y in
                                                             zip(diffs, diffs_shuffle)])
        metrics['slot_algebra_seq_1_4_ratio'] = np.mean([(x + 1e-12) / (x + y + 2e-12) for x, y in
                                                         zip(diffs, diffs_1_4)])
        # clear variables from gpu  torch.cuda.empty_cache()
        del hidden, variables, train_dict, inputs_torch
        torch.cuda.empty_cache()

    if return_all:
        return metrics, time.time() - start_time, {'m_current': m_current,
                                                   'm_n_back': m_n_back,
                                                   'm_loop_slots': m_loop_slots,
                                                   'm_spatial_slots': m_spatial_slots,
                                                   'm_n_back_noslot1': m_n_back_noslot1,
                                                   'm_loop_slots_noslot1': m_loop_slots_noslot1,
                                                   'm_spatial_slots_noslot1': m_spatial_slots_noslot1,
                                                   'cell_to_keep': cell_to_keep,
                                                   # 'variables': variables,
                                                   'cells': None,
                                                   'proj_onto_pca': proj_onto_pca,
                                                   'slot_coords': slot_coords,
                                                   'diff_slots': (diffs, diffs_shuffle, diffs_1_4)
                                                   }

    else:
        return metrics, time.time() - start_time


def phase_discrete(x, div_=None, categories=None):
    x_ = cp.deepcopy(x)
    if div_ is not None:
        assert div_ % 2 == 0  # div_ must be even because really we divide into div_ /2 bits
        if isinstance(x_, list):
            return [phase_discrete(a, div_=div_) for a in x_]
        else:
            x_[np.logical_or(x_ >= (div_ - 1) / div_, x_ < 1 / div_)] = 0
            for i_ in range(1, int(div_ / 2)):
                x_[np.logical_and(x_ >= (2 * i_ - 1) / div_, x_ < (2 * i_ + 1) / div_)] = i_
    else:
        if isinstance(x_, list):
            categories = list(set([item for sublist in x_ for item in sublist.flatten().round(3)]))
            return [phase_discrete(a, categories=categories) for a in x_]
        else:
            if categories is None:
                categories = list(set(x_.flatten().round(3)))
            locs = []
            for i, cat in enumerate(categories):
                locs.append(x_.round(3) == cat)
            for i, loc in enumerate(locs):
                x_[loc] = i

    return x_.astype(int)


def pos_in_loop_slot(pos, slot_id, width):
    # if current position is pos, return position in slot #slot_id
    return (pos - slot_id) % width


def pos_in_spatial_slot(pos, slot_id, width, height):
    # if current position is pos, return position in slot #slot_id
    pos_h = pos // width
    pos_w = pos % width
    slot_id_h = slot_id // width
    slot_id_w = slot_id % width

    new_pos_h = (pos_h + slot_id_h) % height
    new_pos_w = (pos_w + slot_id_w) % width

    return new_pos_h * width + new_pos_w


def slot_algebra_analysis(test_dict, hidden, params, same_index=False):
    diffs, diffs_rand, diffs_1_4 = [], [], []
    index_, index_rand, x_1, x_2, x_3, x_4, x_1_rand, x_2_rand, x_3_rand, skip, loc = None, None, None, None, None, \
        None, None, None, None, None, None
    for i in range(hidden.shape[1]):
        # choose location
        if i % 4 == 0:
            skip = False
            loc = np.random.randint(params.data.max_states)
        elif skip:
            continue
        # find all time-steps when at that location and beyond exploration (i.e. visited all states)
        possibles = np.logical_and.reduce((test_dict.inputs.position[:, i] == loc,
                                           test_dict.inputs.exploration[:, i] == 0,
                                           np.cumsum(test_dict.inputs.exploration[:, i]) == params.data.max_states))
        # do the same for a random sequence position (having visiting every state)
        possibles_rand = np.logical_and(test_dict.inputs.exploration[:, i] == 0,
                                        np.cumsum(test_dict.inputs.exploration[:, i]) == params.data.max_states)
        # choose time-steps
        if i % 4 > 0 and same_index:
            pass
        else:
            try:
                index_ = np.random.choice(np.where(possibles)[0])
                index_rand = np.random.choice(np.where(possibles_rand)[0])
            except ValueError:
                skip = True

        if i % 4 == 0:
            x_1 = hidden[index_, i, :]
            x_1_rand = hidden[index_rand, i, :]
        elif i % 4 == 1:
            x_2 = hidden[index_, i, :]
            x_2_rand = hidden[index_rand, i, :]
        elif i % 4 == 2:
            x_3 = hidden[index_, i, :]
            x_3_rand = hidden[index_rand, i, :]
        elif i % 4 == 3:
            x_4 = hidden[index_, i, :]
            # mean so normalised over number of neurons...
            diffs.append(np.mean((x_4 - (x_1 - x_2 + x_3)) ** 2))
            diffs_rand.append(np.mean((x_4 - (x_1_rand - x_2_rand + x_3_rand)) ** 2))
            diffs_1_4.append(np.mean((x_4 - x_1) ** 2))
    return diffs, diffs_rand, diffs_1_4


class SimpleKerasModel:
    def __init__(self, hidden_dim, num_classes, input_dim):
        super(SimpleKerasModel, self).__init__()
        self.num_classes = num_classes
        self.model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )
        self.opt = keras.optimizers.AdamW()

    def fit(self, x, y):
        batch_size = 256
        epochs = 50
        y = keras.utils.to_categorical(y, self.num_classes)

        self.model.compile(loss="categorical_crossentropy", optimizer=self.opt, metrics=["accuracy"])
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.05, verbose=0)

        return self

    def score(self, x, y):
        y = keras.utils.to_categorical(y, self.num_classes)
        return self.model.evaluate(x, y, verbose=0)[1]


class MultiLogisticRegression:
    def __init__(self, hidden_dim, num_classes, input_dim):
        super(MultiLogisticRegression, self).__init__()
        self.num_classes = num_classes
        self.model = keras.Sequential(
            [
                keras.Input(shape=input_dim),
                layers.Dense(hidden_dim, activation=None),
                layers.Dense(num_classes, activation="sigmoid"),
            ]
        )
        self.opt = keras.optimizers.legacy.Adam()  # JW: NOTE YOU CHANGED THIS...

    def fit(self, x, y):
        batch_size = 256
        epochs = 200
        self.model.compile(loss=keras.losses.BinaryCrossentropy(),
                           optimizer=self.opt,
                           metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)])
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.05, verbose=0)

        return self

    def score(self, x, y):
        return self.model.evaluate(x, y, verbose=0)[1]


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = mutual_info_score(ys[j, :], ys[j, :])
    return h


def histogram_discretize(target, num_bins=20):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized


def mig(m):
    sorted_m = np.sort(m, axis=0)[::-1]
    return np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :]))


def mir(m):
    score = np.max(m, axis=1) / np.sum(m, axis=1)
    min_mir = 1.0 / m.shape[1]
    return (np.mean(score) - min_mir) / (1 - min_mir)


def clean_mi(m, val=20):
    return m[m.sum(axis=1) > m.sum(axis=1).max() / val, :]


def compute_mir(factors, mus, num_bins=20):
    discretized_mus = histogram_discretize(mus, num_bins=num_bins)
    discretized_factors = histogram_discretize(factors, num_bins=num_bins)
    m = discrete_mutual_info(discretized_mus, discretized_factors)  # num_cells x num_factors
    entropy = discrete_entropy(discretized_factors)
    m = m / entropy[None, ...]
    m_cleaned = clean_mi(m)  # only consider cells that have sizable mutual info
    return mir(m_cleaned), m


def xie2022(params, data, load_batch_size, shuffle=False):
    # 'abstract' positions
    pos = data.test_dict.bptt_data.position[:params.data.n_rewards, :]
    # observation  - 'abstract' positions conjunction
    conjunct = (pos * (params.data.o_size - 1) + data.test_dict.bptt_data.observation[:params.data.n_rewards, :]).T
    # if want to randomise
    if shuffle:
        for i in range(conjunct.shape[1]):
            conjunct[:, i] = np.random.permutation(conjunct[:, i])
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(conjunct)
    conjunct_onehot = enc.transform(conjunct).toarray()

    n_train = int(0.8 * load_batch_size)
    x_train = conjunct_onehot[:n_train]
    x_test = conjunct_onehot[n_train:]
    # this is neural data after all observations presented once
    y_train = data.timeseries.inf[params.data.n_rewards - 1][:n_train]
    y_test = data.timeseries.inf[params.data.n_rewards - 1][n_train:]

    import sklearn.linear_model as lm
    func = lm.Lasso(alpha=0.0001)  # lm.LinearRegression()
    clf = func.fit(x_train, y_train)

    print('train score: ', clf.score(x_train, y_train))
    print('test score: ', clf.score(x_test, y_test))
    # subspaces for each position/rank
    coef_split = np.split(clf.coef_.T, params.data.n_rewards)
    pcas = []
    x_pcas = []
    Xs = []
    for i in range(params.data.n_rewards):
        X = coef_split[i]
        Xs.append(X)
        pca = PCA(n_components=2)
        pca.fit(X)
        pcas.append(pca)
        x_pca = pca.transform(X)
        x_pcas.append(x_pca)
        # print(i, pca.explained_variance_ratio_)

    # principle angles between subspaces
    angles = []
    indices = []
    for i in range(params.data.n_rewards):
        for j in range(params.data.n_rewards):
            if i >= j:
                continue
            a = np.matmul(pcas[i].components_, pcas[j].components_.T)
            U, S, V = np.linalg.svd(a, full_matrices=True)
            principles_angles = np.arccos(S) * 180 / np.pi
            angles.append(principles_angles[0])
            indices.append((i, j))

    # size of subspace
    sizes = []
    for i in range(params.data.n_rewards):
        sizes.append(np.abs(x_pcas[i]).mean())

    # controls (not sure why this bit is here, I think it's an artifact of previous stuff...)
    n_train = int(0.5 * load_batch_size)
    x_train_1 = conjunct_onehot[:n_train]
    y_train_1 = data.timeseries.inf[params.data.n_rewards - 1][:n_train]
    x_train_2 = conjunct_onehot[n_train:]
    y_train_2 = data.timeseries.inf[params.data.n_rewards - 1][n_train:]

    func_1 = lm.Lasso(alpha=0.0001)  # lm.LinearRegression()
    clf_1 = func_1.fit(x_train_1, y_train_1)
    # subspaces for each position/rank
    coef_split_1 = np.split(clf_1.coef_.T, params.data.n_rewards)
    pcas_1 = []
    x_pcas_1 = []
    Xs_1 = []
    for i in range(params.data.n_rewards):
        X_1 = coef_split_1[i]
        Xs_1.append(X_1)
        pca_1 = PCA(n_components=2)
        pca_1.fit(X_1)
        pcas_1.append(pca_1)
        x_pca_1 = pca.transform(X_1)
        x_pcas_1.append(x_pca_1)

    func_2 = lm.Lasso(alpha=0.0001)  # lm.LinearRegression()
    clf_2 = func_2.fit(x_train_2, y_train_2)
    # subspaces for each position/rank
    coef_split_2 = np.split(clf_2.coef_.T, params.data.n_rewards)
    pcas_2 = []
    x_pcas_2 = []
    Xs_2 = []
    for i in range(params.data.n_rewards):
        X_2 = coef_split_2[i]
        Xs_2.append(X_2)
        pca_2 = PCA(n_components=2)
        pca_2.fit(X_2)
        pcas_2.append(pca_2)
        x_pca_2 = pca.transform(X_2)
        x_pcas_2.append(x_pca_2)

    # principle angles between subspaces
    angles_control = []
    indices_control = []
    for i in range(params.data.n_rewards):
        a = np.matmul(pcas_1[i].components_, pcas_2[i].components_.T)
        U, S, V = np.linalg.svd(a, full_matrices=True)
        principles_angles = np.arccos(S) * 180 / np.pi
        angles_control.append(principles_angles[0])
        indices_control.append((i, i))

    return pcas, x_pcas, Xs, indices, angles, sizes, angles_control, indices_control


def panichello2021(params, data, load_batch_size, plot_specs, fig_name=''):
    # get neurons
    n_train = int(0.8 * load_batch_size)
    # get neurons at a variety of timesteps before and after cue
    x_train_pre = data.timeseries.inf[params.data.n_rewards - 1][:n_train]
    x_train_pre_1 = data.timeseries.gen[params.data.n_rewards][:n_train]
    x_train_mid = data.timeseries.inf[params.data.n_rewards][:n_train]
    x_train_post = data.timeseries.gen[params.data.n_rewards + 1][:n_train]  # gen as gen is prediction at this timestep

    # these ones don't get used
    x_test_pre = data.timeseries.inf[params.data.n_rewards - 1][n_train:]
    x_test_pre_1 = data.timeseries.gen[params.data.n_rewards][n_train:]
    x_test_mid = data.timeseries.inf[params.data.n_rewards][n_train:]
    x_test_post = data.timeseries.gen[params.data.n_rewards + 1][n_train:]  # gen as gen is prediction at this timestep

    # find indices of correct observation - cue pairs
    # (this is weird due to how I structured inputs. But it's just finding that the cue is for each batch )
    cue = data.test_dict.bptt_data.observation[params.data.n_rewards, :] - (
            params.data.o_size - params.data.n_rewards - 1)
    # what the 'correct' observation is, i.e. the 'top' observation if the cue is 'top'
    obv_correct = data.test_dict.bptt_data.observation[cue, np.arange(cue.shape[0])]

    # observation cue pairs
    obs_cue_pair = [(a, b) for a, b in zip(obv_correct, cue)]
    obs_cue_pair_set = set(obs_cue_pair)
    # get batch index for each pair
    indices_train = [[index for (index, item) in enumerate(obs_cue_pair) if item == a and index < n_train] for a in
                     obs_cue_pair_set]
    indices_test = [[index - n_train for (index, item) in enumerate(obs_cue_pair) if item == a and index >= n_train] for
                    a in obs_cue_pair_set]

    # order these
    sort_index = [i for i, x in sorted(enumerate(list(obs_cue_pair_set)), key=lambda x: x[1])]
    obs_cue_pair_set = list(obs_cue_pair_set)
    obs_cue_pair_set = [obs_cue_pair_set[i] for i in sort_index]
    indices_train = [indices_train[i] for i in sort_index]
    indices_test = [indices_test[i] for i in sort_index]

    assert all([len(x) > 0 for x in indices_train])
    assert all([len(x) > 0 for x in indices_test])

    # make mean neural firing for each condition
    colors = ['mediumturquoise', 'yellowgreen', 'darksalmon', 'orchid', 'c', 'y']
    markers = ['o', 'v']
    subspace = np.array([x[1] for x in list(obs_cue_pair_set)])
    obj = np.array([x[0] for x in list(obs_cue_pair_set)])
    markers_ = np.array([markers[x] for x in subspace])
    colors_ = np.array([colors[x] for x in obj])

    f = plt.figure(figsize=(4 * 4, 20))
    comps_all, cosines = [], []
    for plot_i, (pre_mid_post, name) in enumerate(
            zip([x_train_pre, x_train_pre_1, x_train_mid, x_train_post],
                ['Pre-cue', 'Pre-cue 2', 'Cue', 'Post-cue'])):  # x_train_pre_1
        means_train = []
        for c_o, indices in zip(obs_cue_pair_set, indices_train):
            means_train.append(np.mean(pre_mid_post[indices], axis=0))
        means_train = np.array(means_train)
        means_train = means_train - np.mean(means_train, axis=0)

        # do pca
        pca = PCA(n_components=3)
        pca.fit(means_train)
        means_train_pca = pca.transform(means_train)

        # compute 'plane' for each location (aka cue)
        comps = []
        for loc in range(params.data.n_rewards):
            ind_loc = [k for k, x in enumerate(obs_cue_pair_set) if x[1] == loc]
            means_train_pca_loc = means_train_pca[ind_loc] - np.mean(means_train_pca[ind_loc], axis=0)

            # do pca
            pca = PCA(n_components=2)
            pca.fit(means_train_pca_loc)
            comps.append(pca.components_)
            # print('pre_mid_post:', plot_i, ', loc:', loc, ', explained var:', pca.explained_variance_ratio_)

        comps_all.append(comps)
        normal_0 = np.cross(comps[0][0], comps[0][1])
        normal_1 = np.cross(comps[1][0], comps[1][1])
        cosine = np.sum(normal_0 * normal_1)
        # print('cosine:', cosine)
        cosines.append(np.abs(cosine))

        # plot surface
        ax = plt.subplot(1, 4, plot_i + 1, projection='3d')
        mis, mas = [], []
        for loc in range(params.data.n_rewards):
            # a plane is a*x+b*y+c*z+d=0
            # [a,b,c] is the normal. Thus, we have to calculate
            # d and we're set
            if loc == 0:
                normal = normal_0
                col = 'r'
            else:
                normal = normal_1
                col = 'b'
            ids = np.where(subspace == loc)[0]
            points = means_train_pca[ids]
            center = points.mean(axis=0)
            d = -center.dot(normal)
            X = np.arange(points[:, 0].min(), points[:, 0].max(), (points[:, 0].max() - points[:, 0].min()) / 10)
            Y = np.arange(points[:, 1].min(), points[:, 1].max(), (points[:, 1].max() - points[:, 1].min()) / 10)
            xx, yy = np.meshgrid(X, Y)
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            # print('normal: ', normal)
            # print('min_max_x: ', xx.min(), xx.max())
            # print('min_max_y: ', yy.min(), yy.max())
            # print('min_max_z: ', z.min(), z.max())
            ax.plot_surface(xx, yy, z, alpha=0.15, color='k', linewidth=0)

            mas.append(max([xx.max(), yy.max(), z.max()]))
            mis.append(min([xx.min(), yy.min(), z.min()]))
            # print([xx.max(), yy.max(), z.max()])
            # print([xx.min(), yy.min(), z.min()])

        # plot
        for marker in np.unique(markers_):
            ids = np.where(markers_ == marker)[0]
            ax.scatter(means_train_pca[ids, 0], means_train_pca[ids, 1], means_train_pca[ids, 2], c=colors_[ids], s=120,
                       marker=marker)
            # Draw lines between the points
            num = len(means_train_pca[ids, 0])
            for i in range(num):
                ax.plot(
                    [means_train_pca[ids, 0][i], means_train_pca[ids, 0][(i + 1) % num]],
                    [means_train_pca[ids, 1][i], means_train_pca[ids, 1][(i + 1) % num]],
                    [means_train_pca[ids, 2][i], means_train_pca[ids, 2][(i + 1) % num]],
                    color='black', alpha=0.3)

        mi = min(mis)
        ma = max(mas)
        # print(mi, ma)
        if plot_i == 0:
            ax.view_init(elev=15, azim=30)
        elif plot_i == 3:
            ax.view_init(elev=15, azim=5)
        ax.set_xlim(mi, ma)
        ax.set_ylim(mi, ma)
        ax.set_zlim(mi, ma)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.zaxis.set_major_locator(MaxNLocator(3))
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
            axis.line.set_linewidth(2)
        ax.tick_params(axis='x', labelsize=plot_specs.fontsize)
        ax.tick_params(axis='y', labelsize=plot_specs.fontsize)
        ax.tick_params(axis='z', labelsize=plot_specs.fontsize)

        # ax.set_title(name, fontsize=plot_specs.titlesize)
        # ax.set_xlabel(textwrap.fill('Location-colour PC 1', width=15), fontsize=plot_specs.fontsize, rotation=0,
        #              labelpad=15)
        # ax.set_ylabel(textwrap.fill('Location-colour PC 2', width=15), fontsize=plot_specs.fontsize, rotation=0,
        #              labelpad=15)
        # ax.set_zlabel('Location-colour PC 3', fontsize=plot_specs.fontsize)

    f.savefig('./figures/' + fig_name + ".png", bbox_inches='tight')
    plt.show()
    plt.close('all')

    return cosines
