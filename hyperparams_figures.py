#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""


"""
Fig 1 & 2 : Model Behaviour + Decoding Analyses
"""
default_DB = {
    '    model.model_type = ': ['"EM"', '"WM"'],
    '    model.h_size = ': ['32', '64', '128'],
    '    model.hidden_act = ': ['"tanh"', '"relu"'],
    '    model.transition_type = ': ['"group"', '"conventional_rnn"', '"bio_rnn_add"'],
    '    model.transition_init = ': ['"trunc_norm"'],
    '    model.norm_pi_to_pred = ': ['True'],
    '    train.train_on_visited_states_only = ': ['True', 'False'],
    '    train.hidden_l2_pen = ': ['0.00'],
    '    train.weight_l2_reg_val = ': ['5e-8'],
    '    train.lh_kl_val = ': ['0.00'],
    '    train.amsgrad = ': ['True'],
    '    train.train_iters = ': ['100001'],
    '    train.learning_rate_min = ': ['0.2e-3'],
    '    train.two_dim_output = ': ['False'],
    '    # dummy_line_for_repeat': ['0', '1', '2', '3', '4'],
}
default_data_BD = {'    data.o_size = ': ['10'],
                   '    data.sample_observation_without_replacement = ': ['False'],
                   }
# LoopRandom + LoopRepeat
hyper_dict_BD_lra = {'    data.world_type = ': ['"loop"'],
                     '        par.behaviour_type = ': ['"random"'],
                     '        length = ': ['2', '4', '6', '8', '12',  '16', '20', '24', '28', '32'],
                     }
hyper_dict_BD_lre = {'    data.world_type = ': ['"loop"'],
                     '        par.behaviour_type = ': ['"repeat"'],
                     '        length = ': ['2', '4', '6', '8', '12',  '16', '20', '24', '28', '32'],
                     }
# NBack
hyper_dict_BD_n = {'    data.world_type = ': ['"NBack"'],
                   '        length = ': ['2', '4', '6', '8', '12',  '16', '20', '24', '28', '32'],
                   }
# Rectangle
hyper_dict_BD_r = {'    data.world_type = ': ['"rectangle"'],
                   '        width, height = ': ['2, 2', '2, 3', '3, 3', '3, 4', '4, 4', '4, 5', '5, 5']
                   }

hyper_dict_BD = [hyper_dict_BD_lra, hyper_dict_BD_n, hyper_dict_BD_lre, hyper_dict_BD_r]
hyper_dict_BD = [x | default_data_BD | default_DB for x in hyper_dict_BD]

"""
Fig 3: Slots Algebra + Bio Constraints
"""
default_SB = {
    '    model.model_type = ': ['"WM"'],
    '    model.h_size = ': ['256'],
    '    model.hidden_act = ': ['"tanh"', '"relu"'],
    '    model.transition_type = ': ['"group"', '"conventional_rnn"', '"bio_rnn_add"', '"rnn_add"'],
    '    model.transition_init = ': ['"trunc_norm"'],
    '    model.norm_pi_to_pred = ': ['False'],
    '    train.train_on_visited_states_only = ': ['True'],
    '    train.hidden_l2_pen = ': ['0.00', '0.01'],
    '    train.weight_l2_reg_val = ': ['1e-6'],
    '    train.lh_kl_val = ': ['0.00', '0.01'],
    '    train.amsgrad = ': ['False'],
    '    train.train_iters = ': ['100001'],
    '    train.learning_rate_min = ': ['0.2e-3'],
    '    train.two_dim_output = ': ['False'],
    '    # dummy_line_for_repeat': ['0', '1', '2'],
}
default_data_SB = {'    data.o_size = ': ['10'],
                   '    data.sample_observation_without_replacement = ': ['False'],
                   }
# LoopRandom
hyper_dict_SB_lra = {'    data.world_type = ': ['"loop"'],
                     '        par.behaviour_type = ': ['"random"'],
                     '        length = ': ['4'],
                     }
# LoopRepeat
hyper_dict_SB_lre = {'    data.world_type = ': ['"loop"'],
                     '        par.behaviour_type = ': ['"repeat"'],
                     '        length = ': ['4'],
                     }
# Rectangle
hyper_dict_SB_r = {'    data.world_type = ': ['"rectangle"'],
                   '        width, height = ': ['2, 2'],
                   }

hyper_dict_SB = [hyper_dict_SB_lra, hyper_dict_SB_lre, hyper_dict_SB_r]
hyper_dict_SB = [x | default_data_SB | default_SB for x in hyper_dict_SB]

"""
Fig 4: Controlling Slots
"""
default_CS = {
    '    model.model_type = ': ['"WM"'],
    '    model.h_size = ': ['512'],
    '    model.hidden_act = ': ['"relu"'],
    '    model.transition_type = ': ['"conventional_rnn"'],
    '    model.transition_init = ': ['"trunc_norm"'],
    '    model.norm_pi_to_pred = ': ['True'],
    '    train.train_on_visited_states_only = ': ['True'],
    '    train.hidden_l2_pen = ': ['0.00'],
    '    train.weight_l2_reg_val = ': ['5e-8'],
    '    train.lh_kl_val = ': ['0.00'],
    '    train.amsgrad = ': ['True'],
    '    train.train_iters = ': ['100001'],
    '    train.learning_rate_min = ': ['0.2e-3'],
    '    train.two_dim_output = ': ['False'],
    '    # dummy_line_for_repeat': ['0', '1', '2', '3', '4', '5'],
}
default_data_CS = {'    data.o_size = ': ['10'],
                   }
# LoopVel
hyper_dict_CS_lv = {'    data.world_type = ': ['"loop"'],
                    '        par.behaviour_type = ': ['"2,1,0,-1,0,1"', '"1,1,-1,1,1,0"', '"1,1,-1,1,1,0,1,0,-1,0,1"'],
                    '        length = ': ['7'],
                    '    data.sample_observation_without_replacement = ': ['False'],
                    }
# LoopChunk
hyper_dict_CS_lc = {'    data.world_type = ': ['"loop_chunk"'],
                    '        length = ': ['8', '12', '16'],
                    '    data.sample_observation_without_replacement = ': ['False'],
                    }
# RectVel
hyper_dict_CS_rv = {'    data.world_type = ': ['"rectangle_behave"'],
                    '        par.behaviour_type = ': ['"up,left,down,down,right,right,up,up"'],
                    '        width, height = ': ['3, 3'],
                    '    data.sample_observation_without_replacement = ': ['False'],
                    }
# RectVel
hyper_dict_CS_rc = {'    data.world_type = ': ['"rectangle_chunk"'],
                    '        width, height = ': ['3, 3', '3, 4', '4, 4'],
                    '    data.sample_observation_without_replacement = ': ['False'],
                    }
# LoopDiffSizes
hyper_dict_CS_lds = {'    data.world_type = ': ['"loop_diff_sizes"'],
                     '    data.sample_observation_without_replacement = ': ['False'],
                     }
# LoopDelay
hyper_dict_CS_ld = {'    data.world_type = ': ['"loop_delay"'],
                    '        length = ': ['4'],  # '2', '3', '4'
                    '    data.sample_observation_without_replacement = ': ['True'],
                    "            'delay_max': ": ['6,'],
                    }
# LoopDelay
hyper_dict_CS_lsd = {'    data.world_type = ': ['"loop_same_delay"'],
                     '        length = ': ['2'],  # '2', '3', '4'
                     '    data.sample_observation_without_replacement = ': ['True'],
                     "            'delay_max': ": ['8,'],
                     }

hyper_dict_CS = [hyper_dict_CS_ld]
hyper_dict_CS = [x | default_data_CS | default_CS for x in hyper_dict_CS]

"""
Fig 5: PFC data
"""
default_PFC = {
    '    model.model_type = ': ['"WM"'],
    '    model.h_size = ': ['128'],
    '    model.hidden_act = ': ['"relu"'],
    '    model.transition_type = ': ['"conventional_rnn"'],
    '    model.transition_init = ': ['"orthogonal"'],
    '    model.norm_pi_to_pred = ': ['False'],
    '    train.train_on_visited_states_only = ': ['True'],
    '    train.hidden_l2_pen = ': ['0.0003'],
    '    train.weight_l2_reg_val = ': ['1e-6'],
    '    train.lh_kl_val = ': ['0.00'],
    '    train.amsgrad = ': ['False'],
    '    train.train_iters = ': ['100001'],
    '    train.learning_rate_min = ': ['0.1e-3'],
    '    train.two_dim_output = ': ['True'],
    '    # dummy_line_for_repeat': ['0', '1', '2'],
}
default_data_PFC = {'    data.sample_observation_without_replacement = ': ['True'],
                    }
# LoopVel
hyper_dict_PFC_pa = {'    data.world_type = ': ['"Panichello2021"'],
                     '        length = ': ['2'],
                     '    data.o_size = ': ['4'],
                     }
# NBack
hyper_dict_PFC_xi = {'    data.world_type = ': ['"Xie2022"'],
                     '        length = ': ['3'],
                     '    data.o_size = ': ['6'],
                     }

hyper_dict_PFC = [hyper_dict_PFC_xi, hyper_dict_PFC_pa]
hyper_dict_PFC = [x | default_data_PFC | default_PFC for x in hyper_dict_PFC]

"""
Dataset Creation
"""
# LoopRandom + LoopRepeat
hyper_dict_DS_lr_lr = {
    '    data.world_type = ': ['"loop"'],
    '        par.behaviour_type = ': ['"random"', '"repeat"'],
    '        length = ': ['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '32'],
    '    data.sample_observation_without_replacement = ': ['False'],
    '    data.o_size = ': ['10']
}
# NBack
hyper_dict_DS_nb = {
    '    data.world_type = ': ['"NBack"'],
    '        length = ': ['2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24', '26', '28', '30', '32'],
    '    data.sample_observation_without_replacement = ': ['False'],
    '    data.o_size = ': ['10']
}
# Rectangle
hyper_dict_DS_re = {
    '    data.world_type = ': ['"rectangle"'],
    '        width, height = ': ['1, 2', '2, 2', '2, 3', '3, 3', '3, 4', '4, 4', '4, 5', '5, 5'],
    '    data.sample_observation_without_replacement = ': ['False'],
    '    data.o_size = ': ['10']
}
# LoopVel
hyper_dict_DS_lv = {
    '    data.world_type = ': ['"loop"'],
    '        par.behaviour_type = ': ['"2,1,0,-1,0,1"', '"1,1,-1,1,1,0"', '"1,1,-1,1,1,0,1,0,-1,0,1"'],
    '        length = ': ['7', '5'],
    '    data.sample_observation_without_replacement = ': ['False'],
    '    data.o_size = ': ['10']
}
# RectVel
hyper_dict_DS_rv = {
    '    data.world_type = ': ['"rectangle_behave"'],
    '        par.behaviour_type = ': ['"up,left,down,down,right,right,up,up"'],
    '        width, height = ': ['3, 3'],
    '    data.sample_observation_without_replacement = ': ['False'],
    '    data.o_size = ': ['10']
}
# LoopChunk
hyper_dict_DS_lc = {
    '    data.world_type = ': ['"loop_chunk"'],
    '        length = ': ['4', '6', '8', '12', '16'],
    '    data.sample_observation_without_replacement = ': ['False'],
    '    data.o_size = ': ['10']
}
# RectChunk
hyper_dict_DS_rc = {
    '    data.world_type = ': ['"rectangle_chunk"'],
    '        width, height = ': ['2, 3', '3, 3', '3, 4', '4, 4'],
    '    data.sample_observation_without_replacement = ': ['False'],
    '    data.o_size = ': ['10']
}
# LoopDelay
hyper_dict_DS_ld = {
    '    data.world_type = ': ['"loop_delay"'],
    '        length = ': ['2', '3', '4'],
    '    data.sample_observation_without_replacement = ': ['True', 'False'],
    '    data.o_size = ': ['10'],
    "            'delay_max': ": ['6,'],
}
# LoopSameDelay
hyper_dict_DS_lsd = {
    '    data.world_type = ': ['"loop_same_delay"'],
    '        length = ': ['2', '3', '4'],
    '    data.sample_observation_without_replacement = ': ['True', 'False'],
    '    data.o_size = ': ['10'],
    "            'delay_max': ": ['8,'],
}
# Panichello
hyper_dict_DS_PFC_pa = {
    '    data.world_type = ': ['"Panichello2021"'],
    '        length = ': ['2'],
    '    data.sample_observation_without_replacement = ': ['False', 'True'],
    '    data.o_size = ': ['4']
}
# Xie
hyper_dict_DS_PFC_xi = {
    '    data.world_type = ': ['"Xie2022"'],
    '        length = ': ['3'],
    '    data.sample_observation_without_replacement = ': ['False', 'True'],
    '    data.o_size = ': ['6']
}

hyper_dict_DS = [hyper_dict_DS_lr_lr, hyper_dict_DS_nb, hyper_dict_DS_re, hyper_dict_DS_lv, hyper_dict_DS_ld,
                 hyper_dict_DS_PFC_pa, hyper_dict_DS_PFC_xi]