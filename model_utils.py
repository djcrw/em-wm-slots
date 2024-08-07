#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import torch
import numpy as np
from functools import partial

eps = 1e-8


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        # We trust the dict to init itself better than we can.
        dict.__init__(self, *args, **kwargs)
        # Because of that, we do duplicate work, but it's worth it.
        for k, v in self.items():
            self.__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            # Maintain consistent syntactical behaviour.
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    def __setitem__(self, k, v):
        dict.__setitem__(self, k, DotDict.__convert(v))

    __setattr__ = __setitem__

    def __delattr__(self, k):
        try:
            dict.__delitem__(self, k)
        except KeyError:
            raise AttributeError(
                "'DotDict' object has no attribute '" + str(k) + "'"
            )

    @staticmethod
    def __convert(o):
        """
        Recursively convert `dict` objects in `dict`, `list`, `set`, and
        `tuple` objects to `attrdict` objects.
        """
        if isinstance(o, dict):
            o = DotDict(o)
        elif isinstance(o, list):
            o = list(DotDict.__convert(v) for v in o)
        elif isinstance(o, set):
            o = set(DotDict.__convert(v) for v in o)
        elif isinstance(o, tuple):
            o = tuple(DotDict.__convert(v) for v in o)
        return o

    @staticmethod
    def to_dict(data):
        """
        Recursively transforms a dotted dictionary into a dict
        """
        if isinstance(data, dict):
            data_new = {}
            for k, v in data.items():
                data_new[k] = DotDict.to_dict(v)
            return data_new
        elif isinstance(data, list):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, set):
            return [DotDict.to_dict(i) for i in data]
        elif isinstance(data, tuple):
            return [DotDict.to_dict(i) for i in data]
        else:
            return data


def threshold_torch(x, thresh, thresh_slope=0.1):
    between_thresh = torch.clamp(x, min=-thresh, max=thresh)
    above_thresh = torch.clamp(x - thresh, min=0)
    below_thresh = torch.clamp(x + thresh, max=0)

    return between_thresh + thresh_slope * (above_thresh + below_thresh)


def inputs_2_torch(input_vars, scalings, device='cpu'):
    inputs_dict = DotDict(
        {key: torch.from_numpy(val).type(torch.int32 if 'int' in val.dtype.name else torch.float32).to(device) for
         key, val in input_vars.items()})

    scalings_torch = {}
    for key, val in scalings.items():
        scalings_torch[key] = torch.tensor(val).type(torch.float32).to(device)
    inputs_dict.scalings = scalings_torch

    return inputs_dict


def acc_func_torch(label, pred, axis=1, reduce_mean=True):
    correct = (label == np.argmax(pred, axis=axis)).astype(np.float32)
    return np.mean(correct) if reduce_mean else correct


def gradient_norms(model):
    total_norm = 0
    parameters = [(name, p) for name, p in model.named_parameters() if p.grad is not None and p.requires_grad]
    p_norms = []
    for name, p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        p_norms.append((name, param_norm))
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return p_norms, total_norm


def find_repeat_length(arr):
    for repeat_length in range(1, len(arr) // 2):
        if np.array_equal(arr[:repeat_length], arr[repeat_length:2 * repeat_length]):
            return repeat_length

    return len(set(arr))  # otherwise return number of positions


def compute_accuracies_torch(inputs, preds, pars):
    inputs = torch2numpy(inputs)
    preds = torch2numpy(preds)
    accuracies = DotDict()

    exploration = inputs.exploration[::pars.intermediate_steps + 1, ...]
    if pars.world_type not in ['rectangle', 'rectangle_behave']:
        repeat_lengths = [find_repeat_length(x) for x in inputs.position.T]
        second_loop_indices = [(repeat_length, repeat_length * 2) for repeat_length in repeat_lengths]
        first_predictable_state = [min(np.where(x < 1)[0]) for x in exploration.T]
    else:
        second_loop_indices = [None, None]
        first_predictable_state = None
    names = ['position', 'reward', 'action', 'observation', 'target_o', 'rnn_target_o', 'mem_target_o']
    for name in (names + [x + '_ae' for x in names]):
        if name not in preds.keys():
            continue

        # Remove travelling states
        if name in ['rnn_target_o', 'mem_target_o']:
            inp_ = inputs['target_o'][::pars.intermediate_steps + 1, ...]
        else:
            inp_ = inputs[name.replace('_ae', '')][::pars.intermediate_steps + 1, ...]
        pred_ = preds[name][::pars.intermediate_steps + 1, ...]

        # overall accuracies
        accuracies[name] = acc_func_torch(inp_, pred_, axis=2)
        # overall accuracies not in exploration
        accuracies[name + '_poss_to_predict'] = np.sum(
            (1 - exploration) * acc_func_torch(inp_, pred_, axis=2, reduce_mean=False)) / np.sum(1 - exploration)

        if pars.world_type in ['loop_delay', 'loop_same_delay']:
            # ignore delay bits
            to_keep = inputs.reward[::pars.intermediate_steps + 1, ...]
            accuracies[name + '_nodelay'] = np.sum(
                to_keep * acc_func_torch(inp_, pred_, axis=2, reduce_mean=False)) / np.sum(to_keep)
            to_keep = (1 - exploration) * inputs.reward[::pars.intermediate_steps + 1, ...]
            accuracies[name + '_nodelay_poss_to_predict'] = np.sum(
                to_keep * acc_func_torch(inp_, pred_, axis=2, reduce_mean=False)) / np.sum(to_keep)
            # JW: DO THIS

        if pars.world_type not in ['rectangle', 'rectangle_behave']:
            # accuracies after 1st full loop
            acc = []
            for b in range(pars.batch_size):
                start, stop = second_loop_indices[b]
                acc.append(acc_func_torch(inp_[start:stop, b], pred_[start:stop, b], axis=1))
            accuracies[name + '_second_loop'] = np.mean(acc)

            # accuracies post exploration stage
            for y in [1, 3, 5, 10, 20]:
                acc = []
                for b in range(pars.batch_size):
                    fps = first_predictable_state[b]
                    if fps + y > pred_[:, b, :].shape[0]:
                        continue
                    acc.append(acc_func_torch(inp_[fps:fps + y, b], pred_[fps:fps + y, b, :], axis=1))
                if len(acc) < 1:
                    continue
                accuracies[name + '_' + str(y) + '_post'] = np.mean(acc)

    return accuracies


def make_summaries_torch(inputs, losses, accuracies, scalings, variables, curric_env, env_steps, model,
                         metrics, norms, pars):
    losses = torch2numpy(losses)
    variables = torch2numpy(variables)
    inputs = torch2numpy(inputs)

    summaries = {}
    for key, val in losses.items():
        summaries['losses/' + key] = val

    for key, val in accuracies.items():
        summaries['accuracies/' + key] = val

    for key, val in scalings.items():
        summaries['scalings/' + key] = val

    for key, val in metrics.items():
        summaries['metrics/' + key] = val

    if pars.misc.log_weight_stats:
        for name, norm in norms:
            summaries['gradient_norms/' + name] = norm.detach().cpu().numpy()

        parameters = [(name, p) for name, p in model.named_parameters() if p.grad is not None and p.requires_grad]
        for name, p in parameters:
            summaries['weight_norms/' + name] = p.detach().data.norm(2).cpu().numpy()

    # log memory usage:
    # memory gating behaviour 1st loop vs 2nd loop
    average_firing = np.mean(variables.hidden.inf, axis=(0, 1))
    cell_to_keep = average_firing > np.max(average_firing) / 20
    repeat_lens = np.array([find_repeat_length(x) for x in inputs.position.T])

    def f_mean(mult, arr, a):
        return np.mean(arr[mult * a:(mult + 1) * a])

    def f_max(mult, arr, a):
        return np.max(arr[mult * a:(mult + 1) * a])

    def f_min(mult, arr, a):
        return np.min(arr[mult * a:(mult + 1) * a])

    if pars.misc.log_var_stats:
        for i in range(3):
            if (i + 1) * max(repeat_lens) > variables.hidden.gating_gen_inf.shape[0]:
                continue
            # mean/max/min
            f_mean_ = partial(f_mean, i)
            f_max_ = partial(f_max, i)
            f_min_ = partial(f_min, i)

            gi = variables.hidden.gating_gen_inf[:, :, cell_to_keep].transpose((1, 0, 2))
            summaries['gating/gen_inf_loop_' + str(i) + '_mean'] = np.mean(list(map(f_mean_, gi, repeat_lens)))
            summaries['gating/gen_inf_loop_' + str(i) + '_max'] = np.max(list(map(f_max_, gi, repeat_lens)))
            summaries['gating/gen_inf_loop_' + str(i) + '_min'] = np.min(list(map(f_min_, gi, repeat_lens)))

            if pars.model.external_memory:  # != 'none':
                ms = variables.hidden.mem_store_val.transpose((1, 0))
                summaries['gating/mem_store_val_loop_' + str(i) + '_mean'] = np.mean(
                    list(map(f_mean_, ms, repeat_lens)))
                summaries['gating/mem_store_val_loop_' + str(i) + '_max'] = np.max(list(map(f_max_, ms, repeat_lens)))
                summaries['gating/mem_store_val_loop_' + str(i) + '_min'] = np.min(list(map(f_min_, ms, repeat_lens)))
                mp = variables.hidden.mem_store_prob.transpose((1, 0))
                summaries['gating/mem_store_prob_loop_' + str(i) + '_mean'] = np.mean(
                    list(map(f_mean_, mp, repeat_lens)))
                summaries['gating/mem_store_prob_loop_' + str(i) + '_max'] = np.max(list(map(f_max_, mp, repeat_lens)))
                summaries['gating/mem_store_prob_loop_' + str(i) + '_min'] = np.min(list(map(f_min_, mp, repeat_lens)))

    if pars.misc.log_weight_stats:
        for key, variable in model.named_parameters():
            val = variable.data
            mean = torch.mean(val).detach().cpu().numpy()
            sq = torch.mean(val ** 2).detach().cpu().numpy()
            summaries['weights/' + key + '_max'] = val.max().detach().cpu().numpy()
            summaries['weights/' + key + '_min'] = val.min().detach().cpu().numpy()
            summaries['weights/' + key + '_mean'] = mean
            summaries['weights/' + key + '_sq'] = sq
            summaries['weights/' + key + '_std'] = (sq - mean ** 2) ** 0.5

    var_dict = {'h_inf': variables.hidden.inf,
                'h_gen': variables.hidden.gen,
                }

    if pars.misc.log_var_stats:
        for key, var in var_dict.items():
            summaries['vars/all_' + key + '_min'] = np.min(var)
            summaries['vars/all_' + key + '_max'] = np.max(var)
            e_x = np.mean(var)
            e_x2 = np.mean(var ** 2)
            summaries['vars/all_' + key + '_mean'] = e_x
            summaries['vars/all_' + key + '_sq'] = e_x2
            summaries['vars/all_' + key + '_std'] = (e_x2 - e_x ** 2) ** 0.5
            # we can look at mean, sq, std across population vector or across time/space. Averaging over batch for both
            # across population
            e_x = np.mean(var, axis=2)
            e_x2 = np.mean(var ** 2, axis=2)
            summaries['vars/pop_' + key + '_std'] = np.mean(e_x2 - e_x ** 2) ** 0.5
            # across time-steps
            e_x = np.mean(var, axis=0)
            e_x2 = np.mean(var ** 2, axis=0)
            summaries['vars/time_' + key + '_std'] = np.mean(e_x2 - e_x ** 2) ** 0.5

        if pars.model.external_memory:  # != 'none':
            for key, var in variables.memories.items():
                summaries['mems/' + key + '_min'] = np.mean(np.min(var, axis=(1, 2)))
                summaries['mems/' + key + '_max'] = np.mean(np.max(var, axis=(1, 2)))
                e_x = np.mean(var, axis=(1, 2))
                e_x2 = np.mean(var ** 2, axis=(1, 2))
                summaries['mems/' + key + '_mean'] = np.mean(e_x)
                summaries['mems/' + key + '_sq'] = np.mean(e_x2)
                summaries['mems/' + key + '_std'] = np.mean(e_x2 - e_x ** 2)
            summaries['mems/mem_attention_entropy'] = np.mean(np.sum(
                -variables.memories.attention * np.log(variables.memories.attention + 1e-8), axis=1))

    summaries['extras/new_envs'] = sum(env_steps == 0)
    summaries['extras/av_walk_length'] = np.mean(curric_env.walk_len)
    summaries['extras/data_step_time'] = curric_env.data_step_time
    summaries['extras/forward_step_time'] = curric_env.forward_step_time
    summaries['extras/backward_step_time'] = curric_env.backward_step_time
    summaries['extras/metric_time'] = curric_env.metric_time
    summaries['extras/num_epochs'] = curric_env.num_epochs

    return summaries


def camel_case(name):
    """Converts the given name in snake_case or lowerCamelCase to CamelCase."""
    words = name.split('_')
    return ''.join(word.capitalize() for word in words)


def get_all_keys(value, key_=None):
    """
    Build list of keys for a nested dictionary, so that each value has its own list of nested keys
    """
    key_ = [] if key_ is None else key_
    if not (isinstance(value, dict) or isinstance(value, DotDict)):
        return [[]]
    return [[key] + path for key, val in value.items() for path in get_all_keys(val, key_)]


def nested_set(dic, keys, value):
    """
    Set value of dictionary for a list of nested keys
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def nested_get(dic, keys):
    """
    Get value of dictionary for a list of nested keys
    """
    for key in keys:
        dic = dic[key]
    return dic


def torch2numpy(d):
    if isinstance(d, dict):
        new_dict = DotDict()
        for k, v in d.items():
            new_dict[k] = torch2numpy(v)
        return new_dict
    elif isinstance(d, list):
        return [torch2numpy(x) for x in d]
    elif isinstance(d, tuple):
        return (torch2numpy(x) for x in d)
    elif isinstance(d, torch.Tensor):
        return d.detach().cpu().numpy()
    else:
        return d


def copy_tensor(x):
    if isinstance(x, torch.Tensor):
        return torch.clone(x)
    elif isinstance(x, DotDict):
        return DotDict({key: copy_tensor(value) for key, value in x.items()})
    elif isinstance(x, dict):
        return {key: copy_tensor(value) for key, value in x.items()}
    elif isinstance(x, list):
        return [copy_tensor(y) for y in x]
    else:
        raise ValueError('unsopported type: ' + str(type(x)))


def nested_isnan_inf(x, key=None):
    if isinstance(x, np.ndarray) or isinstance(x, int) or isinstance(x, float) or isinstance(x, np.int64) or isinstance(
            x, np.int32) or isinstance(x, np.float32):
        if np.isnan(x).any() or np.isinf(x).any():
            return True
        else:
            return False
    elif isinstance(x, torch.Tensor):
        return nested_isnan_inf(x.detach().cpu().numpy(), key)
    elif isinstance(x, DotDict) or isinstance(x, dict):
        return {key: nested_isnan_inf(value, key) for key, value in x.items()}
    elif isinstance(x, list):
        return [nested_isnan_inf(y, key) for y in x]
    elif x is None:
        return False
    else:
        raise ValueError('unsopported type: ' + str(type(x)), key)


def is_any_nan_inf(x, current=False):
    if isinstance(x, bool):
        if current:
            return current
        else:
            return x
    elif isinstance(x, DotDict) or isinstance(x, dict):
        return np.asarray([is_any_nan_inf(value) for value in x.values()]).any()
    elif isinstance(x, list):
        return np.asarray([is_any_nan_inf(y) for y in x]).any()
    elif x is None:
        return False
    else:
        raise ValueError('unsopported type: ' + str(type(x)))


def check_inputs_modified(x, y):
    if isinstance(x, torch.Tensor):
        print(torch.sum((x - y) ** 2))
    elif isinstance(x, dict) or isinstance(x, DotDict):
        for key, value in x.items():
            check_inputs_modified(value, y[key])
    elif isinstance(x, list):
        for i, a in enumerate(x):
            check_inputs_modified(a, y[i])
    else:
        print(type(x), type(y))
    return
