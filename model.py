#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import model_utils as mu
import torch
import torch.nn as nn
import math

eps = mu.eps


class AlternationN_Torch(nn.Module):
    def __init__(self, par):
        super(AlternationN_Torch, self).__init__()

        self.par = par
        self.batch_size = None
        self.seq_len = None
        self.seq_pos = None
        self.device = None

        # Hidden Prior
        self.hidden_init = nn.Parameter(torch.zeros((1, self.par.h_size), dtype=torch.float32),
                                        requires_grad=True)

        # Input Embeddings
        if 'observation' in self.par.embedding_inputs:
            self.embed_o = nn.Embedding(self.par.o_size, self.par.embed_size)
        if 'reward' in self.par.embedding_inputs:
            self.embed_r = nn.Embedding(self.par.n_rewards, self.par.embed_size)
        if self.par.use_chunk_action:
            self.embed_chunk_a = nn.Embedding(self.par.n_chunk_actions, self.par.embed_size)
        # MLP of Embeddings

        # RNN Activation
        if self.par.hidden_act == 'none':
            self.activation = nn.Identity()
        elif self.par.hidden_act == 'relu':
            self.activation = nn.ReLU()
        elif self.par.hidden_act[:-5] == 'leaky_relu':
            self.activation = nn.LeakyReLU(float(self.par.hidden_act[-4:]))
        elif self.par.hidden_act == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif self.par.hidden_act == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Activation ' + str(self.par.hidden_act) + ' not implemented yet')

        # RNN Norm
        if self.par.norm_pi_to_pred:
            self.norm_pi_to_pred = nn.LayerNorm(self.par.h_size, elementwise_affine=False)

        # RNN Transition.
        if self.par.transition_type in ['rnn_add', 'conventional_rnn']:
            self.transition = nn.Linear(self.par.h_size, self.par.h_size, bias=True)
        elif self.par.transition_type in ['bio_rnn_add']:
            self.transition_h = nn.Linear(self.par.h_size, self.par.bio_rnn_h_mult * self.par.h_size, bias=True)
            self.transition_g = nn.Linear(self.par.bio_rnn_h_mult * self.par.h_size, self.par.h_size, bias=True)
        elif self.par.transition_type == 'group':
            self.get_transition_vec = nn.Sequential(
                nn.Linear(self.par.v_size, self.par.d_mixed_size, bias=True),
                nn.SiLU(),  # JW: NOTE YOU CHANGED THIS FROM TANH
                nn.Linear(self.par.d_mixed_size, self.par.h_size ** 2, bias=True),
            )
        else:
            raise NotImplementedError('Transition type not implemented yet')

        # RNN Embeddings
        v_embed_size = self.par.bio_rnn_h_mult * self.par.h_size
        self.velocity = nn.Linear(self.par.v_size, v_embed_size, bias=True)

        # RNN Gating (inf/gen)
        self.gating_gen_inf_net = nn.Sequential(
            nn.Linear(self.par.h_size + self.par.h_size, self.par.h_size, bias=True),
            nn.Sigmoid()
        )

        # Predictions
        if 'target_o' in self.par.to_predict:
            self.predict_t = nn.Sequential(
                nn.Linear(self.par.h_size, self.par.o_size, bias=True)
            )

        # Memory Gating (h_pred/mem_pred)
        if self.par.external_memory:
            # add memory choice
            self.add_memory_decision = nn.Sequential(
                nn.Linear(3, 200, bias=True),
                nn.LeakyReLU(0.1),
                nn.Linear(200, 1, bias=True),
            )
            # key / values matrices
            self.wq_k = nn.Linear(self.par.h_size, self.par.key_size, bias=False)
            self.wv = nn.Linear(self.par.h_size, self.par.value_size, bias=False)

            self.norm_after_attention = nn.LayerNorm(self.par.value_size, elementwise_affine=False)

            # init vars
            self.q_k = None
            self.v = None
            self.mem_store = None
            self.inner_prods = 0
            self.attn = 0

        # 2D location of input vectors
        self.readin_loc = nn.Parameter(torch.rand(2), requires_grad=True)  # 0-1
        self.readout_loc = nn.Parameter(torch.rand(2), requires_grad=True)  # 0-1

        _ = self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.par.linear_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.par.embedding_std)

        nn.init.normal_(self.velocity.weight, mean=0.0, std=self.par.embedding_std)
        nn.init.normal_(self.hidden_init, mean=0.0, std=self.par.hidden_init_std)

        if self.par.transition_type in ['rnn_add', 'conventional_rnn']:
            if self.par.transition_init == 'orthogonal':
                nn.init.orthogonal_(self.transition.weight)
            elif self.par.transition_init == 'identity':
                nn.init.zeros_(self.transition.weight)
            elif self.par.transition_init == 'trunc_norm':
                nn.init.trunc_normal_(self.transition.weight, mean=0.0, std=0.001)
            else:
                raise NotImplementedError('Not allowed this transition init')
            if not self.par.add_identity and self.par.transition_init != 'orthogonal':
                self.transition.weight.data = self.transition.weight.data + torch.eye(self.par.h_size)
            nn.init.constant_(self.transition.bias, 0)
        elif self.par.transition_type in ['bio_rnn_add']:
            # initialise transition_h as stacked identities
            mult_copies = torch.tile(torch.eye(self.par.h_size), (self.par.bio_rnn_h_mult, 1))
            self.transition_h.weight.data = mult_copies
            nn.init.constant_(self.transition_h.bias, 0)
            self.transition_h.weight.requires_grad = False
            self.transition_h.bias.requires_grad = False
            # initialise transition_g (as inverse stacked identities)
            if self.par.transition_init == 'orthogonal':
                nn.init.orthogonal_(self.transition_g.weight)
                self.transition_g.weight.data = self.transition_g.weight.data / self.par.bio_rnn_h_mult
            elif self.par.transition_init == 'identity':
                nn.init.zeros_(self.transition_g.weight)
            elif self.par.transition_init == 'trunc_norm':
                nn.init.trunc_normal_(self.transition_g.weight, mean=0.0, std=0.001)
            if not self.par.add_identity and self.par.transition_init != 'orthogonal':
                self.transition_g.weight.data = torch.permute(mult_copies, (1, 0)) / self.par.bio_rnn_h_mult \
                                                + self.transition_g.weight.data
            nn.init.constant_(self.transition_g.bias, 0)
        elif self.par.transition_type == 'group':
            nn.init.trunc_normal_(self.get_transition_vec[2].weight, mean=0.0, std=0.001)
        else:
            raise NotImplementedError('Transition type not implemented yet')

    def forward(self, inputs, device='cpu'):
        # Setup member variables and get dictionaries from input
        rnn_out = self.init_vars(inputs, device=device)
        # embed inputs
        input_to_hidden, vel_embedded = self.inputs_to_hidden(inputs)
        # initialise hidden
        h0 = torch.where(self.seq_pos[:, None] > 0, inputs.hidden,
                         self.rnn_act(self.hidden_init.tile([self.batch_size, 1])))  # remove activation here?
        hidden = h0
        preds = []

        # Run RNN
        mem, mem_retrieved = None, None
        # input_to_hidden[1:], vel_embedded[1:] (i+1)
        for i, (i_to_h, vel) in enumerate(zip(input_to_hidden, vel_embedded)):
            # path integrate (generative)
            pi = self.path_integrate(hidden, vel)
            # apply activation (i.e. real neurons would still be positive)
            pi_to_pred = self.rnn_act(pi)
            # normalise pi to prediction (this is to make RNN activations not want to go v high for better predictions)
            pi_to_pred = self.norm_pi_to_pred(pi_to_pred) if self.par.norm_pi_to_pred else pi_to_pred

            # retrieve memory
            if self.par.external_memory:
                mem_retrieved = self.retrieve_memory(self.wq_k(pi_to_pred), inputs.exploration[:i]) \
                    if i > 0 else torch.zeros_like(pi_to_pred)
                # apply norm (allows memories to be added / removed and retrieved to same extent)
                mem = self.norm_after_attention(mem_retrieved)

            # gate rnn with input (inference)
            hidden, rnn_out['gating_gen_inf'][i] = self.gating_gen_inf(pi, i_to_h)
            # apply activation
            hidden = self.rnn_act(hidden)

            # make predictions
            if self.par.model_type in ['WM']:
                preds.append(self.predict(pi_to_pred))
            elif self.par.model_type in ['EM']:
                preds.append(self.predict(mem))

            # fill in initial stored activations
            rnn_out['gen'][i] = pi
            rnn_out['inf'][i] = hidden

            # add memories
            if self.par.external_memory:
                rnn_out['mem_store_val'][i], rnn_out['mem_store_prob'][i], rnn_out['mem_store_logits'][
                    i] = self.add_memory(hidden, i_to_h, mem_retrieved, seq_pos=i, exploration=inputs.exploration[i])

        # make predictions
        predictions = {**{key: torch.stack([x[0][key] for x in preds], dim=0) for key in preds[0][0].keys()}}
        logits = {**{key: torch.stack([x[1][key] for x in preds], dim=0) for key in preds[0][1].keys()}}
        # collect variables
        variable_dict = mu.DotDict(
            {'hidden': rnn_out,
             'pred': predictions,
             'logits': logits,
             'memories': {'inner_prods': self.inner_prods,
                          'attention': self.attn,
                          } if self.par.external_memory else None
             })
        # Collate g for re-input to model
        re_input_dict = {'hidden': rnn_out['inf'][-1], }

        return variable_dict, re_input_dict

    def rnn_act(self, x):
        x = self.activation(x)
        return mu.threshold_torch(x, self.par.hidden_thresh, thresh_slope=self.par.hidden_thresh_alpha)

    def predict(self, hidden, extra_name=''):
        preds = {}
        logits = {}
        if 'target_o' in self.par.to_predict:
            preds['target_o'], logits['target_o'] = self.f_t(hidden)

        preds = {key + extra_name: val for key, val in preds.items()}
        logits = {key + extra_name: val for key, val in logits.items()}

        return preds, logits

    def f_r(self, hidden):
        r_logits = self.predict_r(hidden)
        r = nn.Softmax(dim=1)(r_logits)

        return r, r_logits

    def f_p(self, hidden):
        p_logits = self.predict_p(hidden)
        p = nn.Softmax(dim=1)(p_logits)

        return p, p_logits

    def f_a(self, hidden):
        a_logits = self.predict_a(hidden)
        a = nn.Softmax(dim=1)(a_logits)

        return a, a_logits

    def f_o(self, hidden, rnn_or_mem=None):
        if rnn_or_mem == 'rnn':
            o_logits = self.predict_o_rnn(hidden)
        elif rnn_or_mem == 'mem':
            o_logits = self.predict_o_mem(hidden)
        else:
            o_logits = self.predict_o(hidden)
        o = nn.Softmax(dim=1)(o_logits)

        return o, o_logits

    def f_t(self, hidden):
        t_logits = self.predict_t(hidden)
        t = nn.Softmax(dim=1)(t_logits)

        return t, t_logits

    def path_integrate(self, hidden, velocity_embedded):
        if self.par.transition_type == 'rnn_add':
            pi = self.transition(hidden) + velocity_embedded
        elif self.par.transition_type == 'bio_rnn_add':
            pi = self.transition_g(self.rnn_act(self.transition_h(hidden) + velocity_embedded))
        elif self.par.transition_type == 'conventional_rnn':
            pi = self.transition(hidden)
        elif self.par.transition_type == 'group':
            transition_mat = velocity_embedded.reshape((-1, self.par.h_size, self.par.h_size))
            pi = torch.squeeze(torch.matmul(transition_mat, hidden[..., None]))
        else:
            raise NotImplementedError('Transition type not implemented yet')

        if self.par.add_identity:
            pi = pi + hidden

        return pi

    def gating_gen_inf(self, path_int, inp):
        sigma = self.gating_gen_inf_net(torch.cat([path_int, inp], dim=1))
        return sigma * path_int + (1.0 - sigma) * inp, sigma

    def gating_pi_mem(self, path_int, mem):
        sigma = self.gating_pi_mem_net(torch.cat([path_int, mem], dim=1))
        return sigma * path_int + (1.0 - sigma) * mem, sigma

    def gating_pred_h_pred_mem(self, h, mem):
        h_pred, h_logits = self.f_o(h, rnn_or_mem='rnn')
        mem_pred, mem_logits = self.f_o(mem, rnn_or_mem='mem')

        sigma = self.gating_pred_h_pred_mem_net(torch.cat([h, mem], dim=1))

        logits = {'target_o': sigma * h_logits + (1.0 - sigma) * mem_logits,
                  'rnn_target_o': h_logits,
                  'mem_target_o': mem_logits, }
        preds = {'target_o': nn.Softmax(dim=1)(logits['target_o']),
                 'rnn_target_o': h_pred,
                 'mem_target_o': mem_pred, }

        rnn_mem_pred_contrib = (h_pred ** 2).sum(dim=1) / ((h_pred ** 2).sum(dim=1) + (mem_pred ** 2).sum(dim=1) + eps)
        rnn_mem_pred_var = torch.std(h_pred, dim=1) / (torch.std(h_pred, dim=1) + torch.std(mem_pred, dim=1) + eps)
        rnn_mem_logits_var = torch.std(h_logits, dim=1) / (
                torch.std(h_logits, dim=1) + torch.std(mem_logits, dim=1) + eps)

        return (preds, logits), sigma, rnn_mem_pred_contrib, rnn_mem_pred_var, rnn_mem_logits_var

    def init_vars(self, inputs, device='cpu'):
        self.batch_size = inputs.reward.shape[1]
        self.seq_len = inputs.reward.shape[0]
        self.seq_pos = inputs.seq_index * self.seq_len
        self.device = device

        if self.par.external_memory:
            self.q_k = torch.zeros((self.batch_size, 0, self.par.key_size)).to(self.device)
            self.v = torch.zeros((self.batch_size, self.par.value_size, 0)).to(self.device)
            self.mem_store = torch.zeros((self.batch_size, 0)).to(self.device)

        return {'inf': torch.zeros((self.seq_len, self.batch_size, self.par.h_size)).to(self.device),
                'gen': torch.zeros((self.seq_len, self.batch_size, self.par.h_size)).to(self.device),
                'gating_gen_inf': torch.ones((self.seq_len, self.batch_size, self.par.h_size)).to(self.device),
                'gating_pi_mem': torch.ones((self.seq_len, self.batch_size, self.par.h_size)).to(self.device),
                'gating_pred_h_pred_mem': torch.ones((self.seq_len, self.batch_size, 1)).to(self.device),
                'rnn_mem_pred_contrib': torch.ones((self.seq_len, self.batch_size)).to(self.device),
                'rnn_mem_pred_var': torch.ones((self.seq_len, self.batch_size)).to(self.device),
                'rnn_mem_logits_var': torch.ones((self.seq_len, self.batch_size)).to(self.device),
                'mem_store_val': torch.zeros((self.seq_len, self.batch_size)).to(self.device),
                'mem_store_prob': torch.zeros((self.seq_len, self.batch_size)).to(self.device),
                'mem_store_logits': torch.zeros((self.seq_len, self.batch_size)).to(self.device),
                }

    def inputs_to_hidden(self, inputs):
        # embed inputs
        e_ins = []
        if 'position' in self.par.embedding_inputs:
            e_ins.append(self.embed_p(inputs.position.reshape(-1)).reshape((self.seq_len, self.batch_size, -1)))
        if 'reward' in self.par.embedding_inputs:
            e_ins.append(self.embed_r(inputs.reward.reshape(-1)).reshape((self.seq_len, self.batch_size, -1)))
        if 'observation' in self.par.embedding_inputs:
            e_ins.append(self.embed_o(inputs.observation.reshape(-1)).reshape((self.seq_len, self.batch_size, -1)))
        # account for travelling - i.e. no info when travelling
        embeds = torch.stack(e_ins, dim=0).sum(dim=0) * (1.0 - inputs.travelling[..., None])
        # pass embeddings through MLP
        input_to_hidden = embeds

        # embed velocity
        if self.par.transition_type in ['conventional_rnn']:
            # want velocity at time-step t to be current velocity (i.e. vel to next loc, not vel to current loc)
            inputs.velocity = torch.roll(inputs.velocity.detach(), -1, dims=0)
        if self.par.transition_type == 'group':
            if not self.par.use_velocity:
                inputs.velocity = torch.ones_like(inputs.velocity.detach())
            vel_embedded = self.get_transition_vec(inputs.velocity.reshape((-1, self.par.v_size))).reshape(
                (self.seq_len, self.batch_size, -1))
        else:
            vel_embedded = self.velocity(inputs.velocity.reshape((-1, self.par.v_size))).reshape(
                (self.seq_len, self.batch_size, -1))
            if not self.par.use_velocity:
                vel_embedded = torch.zeros_like(vel_embedded.detach())

        if self.par.transition_type in ['conventional_rnn']:
            input_to_hidden = input_to_hidden + vel_embedded
        if self.par.use_chunk_action:
            inputs.chunk_action = torch.roll(inputs.chunk_action.detach(), -1, dims=0)
            c_a = self.embed_chunk_a(inputs.chunk_action.reshape(-1)).reshape((self.seq_len, self.batch_size, -1))
            input_to_hidden = input_to_hidden + c_a

        return input_to_hidden, vel_embedded

    def add_memory(self, hidden, i_2_h, mem_retrieved, seq_pos=0, exploration=torch.zeros(0)):
        key_to_add = self.wq_k(hidden.clone())
        value_to_add = self.wv(hidden.clone()) - mem_retrieved.clone()

        prob = torch.ones_like(exploration)
        logits = torch.zeros((self.batch_size, 1, 1)).to(self.device)
        mem_store_val = prob
        self.mem_store = torch.concatenate([self.mem_store, mem_store_val[:, None]], dim=1)
        self.q_k = torch.concatenate([self.q_k, mem_store_val[:, None, None] * key_to_add[:, None, :]], dim=1)
        self.v = torch.concatenate([self.v, mem_store_val[:, None, None] * value_to_add[:, :, None]], dim=2)

        return torch.squeeze(mem_store_val), torch.squeeze(prob), torch.squeeze(logits)

    def retrieve_memory(self, query, exploration):
        # self attention
        inner_prods = (self.q_k @ query[..., None]) / (self.par.key_size ** 0.5)
        inner_prods = inner_prods - (self.mem_store[..., None] - 1) * 100.00

        self.inner_prods = inner_prods
        # change temperature parameter depending on number of memories (leave as is as unused memories are 0s)
        num_mem = exploration.sum(axis=0).detach()[:, None, None]
        beta = 1.0 + torch.log(torch.clamp(num_mem, min=2.0) - 1.0)
        inner_prods = inner_prods * beta
        # make numerically stable
        inner_prods = inner_prods - inner_prods.max(dim=1, keepdim=True)[0]
        attention = nn.Softmax(dim=1)(inner_prods)
        self.attn = attention

        return torch.squeeze(self.v @ attention)


def compute_losses_torch(model_in, model_out, model, par, device='cpu', world_type=None):
    ce_loss = nn.CrossEntropyLoss(reduction='none')

    seq_len = model_in.position.shape[0]
    batch_size = model_in.position.shape[1]

    norm = 1.0 / (seq_len * batch_size)
    norm_p2p = 1.0 / (seq_len * batch_size)
    possible_to_predict = 1.0
    if par.train_on_visited_states_only:
        possible_to_predict = (1.0 - model_in.exploration) * (1.0 - model_in.travelling)
        if torch.sum(possible_to_predict) > 0.0:
            norm_p2p = 1.0 / torch.sum(possible_to_predict)

    losses = mu.DotDict()
    cost_all = 0.0

    if 'target_o' in par.which_costs:
        if world_type in ['Panichello2021', 'Xie2022'] and par.two_dim_output:
            def number_to_coordinate(num, max_size, pad, device='cpu'):
                num_out = torch.zeros(num.shape[0], num.shape[1], pad).to(device)
                angle = num * 2 * torch.tensor(math.pi) / max_size
                num_out[:, :, 0] = torch.cos(angle)
                num_out[:, :, 1] = torch.sin(angle)
                return num_out

            if world_type == 'Panichello2021':
                o_size = model_out.logits.target_o.shape[2] - model_out.logits.target_o.shape[0] + 1
            else:
                o_size = model_out.logits.target_o.shape[2] - 1
            t_n = number_to_coordinate(model_in.target_o.type(torch.long), o_size, model_out.logits.target_o.shape[2],
                                       device=device)
            lo_target = torch.sum(torch.sum((t_n - model_out.logits.target_o) ** 2, 2) * possible_to_predict) * norm_p2p
        else:
            lo_target = torch.sum(ce_loss(torch.permute(model_out.logits.target_o, (0, 2, 1)),
                                          model_in.target_o.type(torch.long)) * possible_to_predict) * norm_p2p
        losses.lo_target = lo_target
        cost_all += lo_target
    # rnn regularisers
    if 'hidden_kl' in par.which_costs:
        lh_kl = torch.sum(
            torch.sum((model_out.hidden.inf - model_out.hidden.gen) ** 2, 2) * possible_to_predict) * norm_p2p
        losses.lh_kl = lh_kl * par.lh_kl_val
        cost_all += lh_kl * par.lh_kl_val
        losses.lh_kl_unscaled = lh_kl
    if 'hidden_l2' in par.which_costs:
        lh_l2 = torch.sum(torch.sum(model_out.hidden.inf ** 2, 2)) * norm
        cost_all += lh_l2 * par.hidden_l2_pen
        losses.lh_l2 = lh_l2 * par.hidden_l2_pen
        losses.lh_l2_unscaled = lh_l2
    # weight regularisers
    if 'weight_l2' in par.which_costs:
        weight_l2 = 0.0
        for name, p in model.named_parameters():
            if 'weight' in name:
                weight_l2 += (p ** 2).sum()
        losses.weight_l2 = weight_l2 * par.weight_l2_reg_val
        losses.weight_l2_unscaled = weight_l2

    losses.train_loss = cost_all

    return losses
