#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import numpy as np
import copy as cp
import itertools


# from line_profiler_pycharm import profile


class Environment:
    def __init__(self, params, width, height):
        super(Environment, self).__init__()
        self.par = params
        self.width = width
        self.height = height
        self.n_actions = self.par.env.n_actions
        self.n_rewards = self.par.n_rewards
        self.rels = self.par.env.rels
        self.walk_len = None
        self.reward_value = 1.0
        self.observation_locations = None
        self.reward_locations = None
        self.start_state, self.adj, self.tran, self.states_mat = None, None, None, None


class RectangleRewards(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width, height)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states(width, height):
        return width * height

    def world(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        states = int(self.width * self.height)
        adj = np.zeros((states, states))

        for i in range(states):
            # stay still
            if self.par.env.stay_still:
                adj[i, i] = 1
            # up - down
            if i + self.width < states:
                adj[i, i + self.width] = 1
                adj[i + self.width, i] = 1
                # left - right
            if np.mod(i, self.width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1

            if self.par.torus and np.mod(i, self.width) == 0:
                adj[i, i + self.width - 1] = 1
                adj[i + self.width - 1, i] = 1

            if self.par.torus and int(i / self.width) == 0:
                adj[i, i + states - self.width] = 1
                adj[i + states - self.width, i] = 1

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran

        # choose reward locations
        self.reward_locations = []
        possible_reward_locs = [i for i in range(self.n_states)]
        for i in range(self.n_rewards):
            reward_location = np.random.choice(possible_reward_locs)
            self.reward_locations.append(reward_location)
            possible_reward_locs.remove(reward_location)

        if self.par.start_at_first_reward:
            self.start_state = self.reward_locations[0]
        else:
            self.start_state = np.random.choice(self.n_states)

    def relation(self, s1, s2):
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == self.width or diff == -self.width * (self.height - 1):  # down
            rel_type = 'down'
        elif diff == -self.width or diff == self.width * (self.height - 1):  # up
            rel_type = 'up'
        elif diff == -1 or diff == (self.width - 1):  # left
            rel_type = 'left'
        elif diff == 1 or diff == -(self.width - 1):  # right
            rel_type = 'right'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = [i for i in range(self.par.o_size - 1 if self.par.non_reward_are_delays_input else self.par.o_size)]
        for i in range(self.n_states):
            if self.par.non_reward_are_delays_input and i not in self.reward_locations:
                states_vec[i] = self.par.o_size - 1
                continue
            if self.par.observation_is_position:
                states_vec[i] = i
            else:
                states_vec[i] = np.random.choice(choices)
                if self.par.sample_observation_without_replacement:
                    choices.remove(states_vec[i])

        self.states_mat = states_vec.astype(int)

    def walk(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        reward_visit = np.zeros((time_steps, self.n_rewards))
        reward = np.zeros(time_steps, dtype=np.int16)
        action = np.zeros(time_steps, dtype=np.int16)
        exploration = np.ones(time_steps, dtype=np.int16)
        cum_reward = np.zeros(self.n_rewards)
        steps_between_rewards = np.zeros(time_steps, dtype=np.int16)
        phase_velocity = np.zeros(time_steps, dtype=np.float32)
        velocity = np.zeros((time_steps, self.par.env.dim_space), dtype=np.float32)
        current_angle = np.random.uniform(-np.pi, np.pi)
        observation = np.zeros(time_steps, dtype=np.int16)
        target_o = np.zeros(time_steps, dtype=np.int16)
        goal_position = np.zeros(time_steps, dtype=np.int16)
        goal_observation = np.zeros(time_steps, dtype=np.int16)

        # consider rectangular environment. if move +1=up, -1=down, +length=right, -length=left.
        if self.height * self.width != len(self.adj):
            raise ValueError('incorrect height/width : height * width not equal to number of states')

        next_reward_id = 0
        total_rewards = 0

        position[0] = int(self.start_state)
        observation[0] = self.states_mat[position[0]]
        if self.par.non_reward_are_delays_target:
            target_o[0] = observation[0] if position[0] == self.reward_locations[
                next_reward_id] else self.par.o_size - 1
        else:
            target_o[0] = observation[0]
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        action[0] = 0

        # detect if at a rewarded location
        if position[0] == self.reward_locations[next_reward_id]:
            reward[0] = 1
            cum_reward[next_reward_id] += 1
            reward_visit[0, next_reward_id] = cum_reward[next_reward_id]
            total_rewards += 1
            next_reward_id = (next_reward_id + 1) % self.n_rewards

        goal_position[0] = self.reward_locations[next_reward_id]
        goal_observation[0] = self.states_mat[self.reward_locations[next_reward_id]]
        len_loop = 0
        steps_between_reward = 1
        for i in range(time_steps - 1):

            if total_rewards >= self.n_rewards:
                exploration[i + 1] = 0

            if self.par.repeat_path and total_rewards > self.n_rewards:
                # here repeat exact path if already done one full loop
                position[i + 1] = position[i + 1 - len_loop]
                steps_between_rewards[i + 1] = steps_between_rewards[i + 1 - len_loop]
            else:
                # only add if have found 1st reward already
                if total_rewards > 0:
                    len_loop += 1
                available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
                # head towards objects, or in straight lines
                if total_rewards >= self.n_rewards or not self.par.agent_initial_exploration:
                    next_reward_state = self.reward_locations[next_reward_id]
                    # go direct to next reward
                    new_poss_pos = self.go_to_closest_reward(next_reward_state, available)
                elif self.par.env.bias_type == 'angle':
                    new_poss_pos, current_angle = self.move_straight_bias(current_angle, position[i], available)
                else:
                    new_poss_pos = np.random.choice(available)

                if self.adj[position[i], new_poss_pos] == 1:
                    position[i + 1] = new_poss_pos
                else:
                    position[i + 1] = int(cp.deepcopy(position[i]))

            relation_taken, rel_type = self.relation(position[i], position[i + 1])
            action[i + 1] = relation_taken
            velocity[i + 1, :] = self.par.env.velocities[relation_taken]
            # stay still is just a set of zeros

            observation[i + 1] = self.states_mat[position[i + 1]]
            if self.par.non_reward_are_delays_target:
                target_o[i + 1] = observation[i + 1] if position[i + 1] == self.reward_locations[
                    next_reward_id] else self.par.o_size - 1
            else:
                target_o[i + 1] = observation[i + 1]

            # detect if at a rewarded location
            if position[i + 1] == self.reward_locations[next_reward_id]:
                reward[i + 1] = 1
                cum_reward[next_reward_id] += 1
                reward_visit[i + 1, next_reward_id] = cum_reward[next_reward_id]
                total_rewards += 1
                next_reward_id = (next_reward_id + 1) % self.n_rewards
                # update steps between this reward and last
                for j in range(steps_between_reward):
                    steps_between_rewards[i + 1 - j] = steps_between_reward
                steps_between_reward = 0
            goal_position[i + 1] = self.reward_locations[next_reward_id]
            goal_observation[i + 1] = self.states_mat[self.reward_locations[next_reward_id]]
            steps_between_reward += 1
        phase_velocity[1:] = 1. / steps_between_rewards[1:]

        walk_data = {'position': position,
                     'action': action,
                     'reward': reward,
                     'exploration': exploration,
                     'steps_between_rewards': steps_between_rewards,
                     'phase_velocity': phase_velocity,
                     'velocity': velocity,
                     'observation': observation,
                     'phase': np.cumsum(phase_velocity) % 1.0,
                     'goal_position': goal_position,
                     'goal_observation': goal_observation,
                     'target_o': target_o,
                     'chunk_action': np.zeros(time_steps, dtype=np.int16),
                     }

        return walk_data

    def move_straight_bias(self, current_angle, position, available):
        # angle is allo-centric
        # from available position - find distance and angle from current pos
        diff_angle_min = np.pi / 4
        angles = [self.angle_between_states_square(position, x) if x != position else 10000 for x in available]
        # find angle closest to current angle
        a_diffs = [np.abs(a - current_angle) for a in angles]
        a_diffs = [a if a < np.pi else np.abs(2 * np.pi - a) for a in a_diffs]

        angle_diff = np.min(a_diffs)

        if angle_diff < diff_angle_min:
            a_min_index = np.where(a_diffs == angle_diff)[0][0]
            angle = current_angle
        else:  # hit a wall - then do random non stationary choice
            p_angles = [1 if a < 100 else 0.000001 for a in angles]
            a_min_index = np.random.choice(np.arange(len(available)), p=np.asarray(p_angles) / sum(p_angles))
            angle = angles[a_min_index]

        new_poss_pos = int(available[a_min_index])

        angle += np.random.uniform(-self.par.env.angle_bias_change, self.par.env.angle_bias_change)
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # keep between +- pi

        if np.random.rand() > self.par.env.direc_bias:
            p = self.tran[int(position), available]
            new_poss_pos = np.random.choice(available, p=p)

        return new_poss_pos, angle

    def angle_between_states_square(self, s1, s2):
        x1 = s1 % self.width
        x2 = s2 % self.width

        y1 = np.floor(s1 / self.width)
        y2 = np.floor(s2 / self.width)

        angle = np.arctan2(y1 - y2, x2 - x1)

        return angle

    def go_to_closest_reward(self, position, available):
        distances = np.array([self.distance_between_states(position, x) for x in available])
        min_indices = [a for a, x in enumerate(distances) if x == distances.min()]

        return available[
            np.random.choice(min_indices)] if np.random.rand() < self.par.env.correct_step else np.random.choice(
            available)

    def distance_between_states(self, s1, s2):
        x1 = s1 % self.width
        x2 = s2 % self.width

        y1 = np.floor(s1 / self.width)
        y2 = np.floor(s2 / self.width)

        distance = np.abs(x2 - x1) + np.abs(y2 - y1)

        return distance

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xs = xs.flatten() - (self.width - 1) / 2
        ys = - (ys.flatten() - (self.height - 1) / 2)

        if cells is not None:
            cell_prepared = cp.deepcopy(cells).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class Rectangle(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width, height)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states(width, height):
        return width * height

    def world(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        states = int(self.width * self.height)
        adj = np.zeros((states, states))

        for i in range(states):
            # stay still
            if self.par.env.stay_still:
                adj[i, i] = 1
            # up - down
            if i + self.width < states:
                adj[i, i + self.width] = 1
                adj[i + self.width, i] = 1
                # left - right
            if np.mod(i, self.width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1

            if self.par.torus and np.mod(i, self.width) == 0:
                adj[i, i + self.width - 1] = 1
                adj[i + self.width - 1, i] = 1

            if self.par.torus and int(i / self.width) == 0:
                adj[i, i + states - self.width] = 1
                adj[i + states - self.width, i] = 1

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran

        # always start at middle-ish state
        self.start_state = ((self.height // 2) * self.width) + self.width // 2

    def relation(self, s1, s2):
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == self.width or diff == -self.width * (self.height - 1):  # down
            rel_type = 'down'
        elif diff == -self.width or diff == self.width * (self.height - 1):  # up
            rel_type = 'up'
        elif diff == -1 or diff == (self.width - 1):  # left
            rel_type = 'left'
        elif diff == 1 or diff == -(self.width - 1):  # right
            rel_type = 'right'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = [i for i in range(self.par.o_size)]
        for i in range(self.n_states):
            states_vec[i] = np.random.choice(choices)
            if self.par.sample_observation_without_replacement:
                choices.remove(states_vec[i])

        self.states_mat = states_vec.astype(int)

    def walk(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        time_steps = self.walk_len
        position = np.zeros(time_steps, dtype=np.int16)
        reward = np.zeros(time_steps, dtype=np.int16)
        action = np.zeros(time_steps, dtype=np.int16)
        exploration = np.ones(time_steps, dtype=np.int16)
        steps_between_rewards = np.zeros(time_steps, dtype=np.int16)
        phase_velocity = np.zeros(time_steps, dtype=np.float32)
        velocity = np.zeros((time_steps, self.par.env.dim_space), dtype=np.float32)
        current_angle = np.random.uniform(-np.pi, np.pi)
        observation = np.zeros(time_steps, dtype=np.int16)
        target_o = np.zeros(time_steps, dtype=np.int16)
        goal_position = np.zeros(time_steps, dtype=np.int16)
        goal_observation = np.zeros(time_steps, dtype=np.int16)

        # consider rectangular environment. if move +1=up, -1=down, +length=right, -length=left.
        if self.height * self.width != len(self.adj):
            raise ValueError('incorrect height/width : height * width not equal to number of states')

        position[0] = int(self.start_state)
        observation[0] = self.states_mat[position[0]]
        target_o[0] = observation[0]
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        action[0] = 0

        visited_positions = [position[0]]
        for i in range(time_steps - 1):
            available = np.where(self.tran[int(position[i]), :] > 0)[0].astype(int)
            if self.par.env.bias_type == 'angle':
                new_poss_pos, current_angle = self.move_straight_bias(current_angle, position[i], available)
            else:
                new_poss_pos = np.random.choice(available)

            if self.adj[position[i], new_poss_pos] == 1:
                position[i + 1] = new_poss_pos
            else:
                position[i + 1] = int(cp.deepcopy(position[i]))

            if position[i + 1] in visited_positions:
                exploration[i + 1] = 0
            else:
                visited_positions.append(position[i + 1])

            relation_taken, rel_type = self.relation(position[i], position[i + 1])
            action[i + 1] = relation_taken
            velocity[i + 1, :] = self.par.env.velocities[relation_taken]
            # stay still is just a set of zeros

            observation[i + 1] = self.states_mat[position[i + 1]]
            target_o[i + 1] = observation[i + 1]

        walk_data = {'position': position,
                     'action': action,
                     'reward': reward,
                     'exploration': exploration,
                     'steps_between_rewards': steps_between_rewards,
                     'phase_velocity': phase_velocity,
                     'velocity': velocity,
                     'observation': observation,
                     'phase': np.cumsum(phase_velocity) % 1.0,
                     'goal_position': goal_position,
                     'goal_observation': goal_observation,
                     'target_o': target_o,
                     'chunk_action': np.zeros(time_steps, dtype=np.int16),
                     }

        return walk_data

    def move_straight_bias(self, current_angle, position, available):
        # angle is allo-centric
        # from available position - find distance and angle from current pos
        diff_angle_min = np.pi / 4
        angles = [self.angle_between_states_square(position, x) if x != position else 10000 for x in available]
        # find angle closest to current angle
        a_diffs = [np.abs(a - current_angle) for a in angles]
        a_diffs = [a if a < np.pi else np.abs(2 * np.pi - a) for a in a_diffs]

        angle_diff = np.min(a_diffs)

        if angle_diff < diff_angle_min:
            a_min_index = np.where(a_diffs == angle_diff)[0][0]
            angle = current_angle
        else:  # hit a wall - then do random non stationary choice
            p_angles = [1 if a < 100 else 0.000001 for a in angles]
            a_min_index = np.random.choice(np.arange(len(available)), p=np.asarray(p_angles) / sum(p_angles))
            angle = angles[a_min_index]

        new_poss_pos = int(available[a_min_index])

        angle += np.random.uniform(-self.par.env.angle_bias_change, self.par.env.angle_bias_change)
        angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi  # keep between +- pi

        if np.random.rand() > self.par.env.direc_bias:
            p = self.tran[int(position), available]
            new_poss_pos = np.random.choice(available, p=p)

        return new_poss_pos, angle

    def angle_between_states_square(self, s1, s2):
        x1 = s1 % self.width
        x2 = s2 % self.width

        y1 = np.floor(s1 / self.width)
        y2 = np.floor(s2 / self.width)

        angle = np.arctan2(y1 - y2, x2 - x1)

        return angle

    def go_to_closest_reward(self, position, available):
        distances = np.array([self.distance_between_states(position, x) for x in available])
        min_indices = [a for a, x in enumerate(distances) if x == distances.min()]

        return available[
            np.random.choice(min_indices)] if np.random.rand() < self.par.env.correct_step else np.random.choice(
            available)

    def distance_between_states(self, s1, s2):
        x1 = s1 % self.width
        x2 = s2 % self.width

        y1 = np.floor(s1 / self.width)
        y2 = np.floor(s2 / self.width)

        distance = np.abs(x2 - x1) + np.abs(y2 - y1)

        return distance

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xs = xs.flatten() - (self.width - 1) / 2
        ys = - (ys.flatten() - (self.height - 1) / 2)

        if cells is not None:
            cell_prepared = cp.deepcopy(cells).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class RectangleBehave(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width, height)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states(width, height):
        return width * height

    def world(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        states = int(self.width * self.height)
        adj = np.zeros((states, states))

        for i in range(states):
            # stay still
            if self.par.env.stay_still:
                adj[i, i] = 1
            # up - down
            if i + self.width < states:
                adj[i, i + self.width] = 1
                adj[i + self.width, i] = 1
                # left - right
            if np.mod(i, self.width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1

            if self.par.torus and np.mod(i, self.width) == 0:
                adj[i, i + self.width - 1] = 1
                adj[i + self.width - 1, i] = 1

            if self.par.torus and int(i / self.width) == 0:
                adj[i, i + states - self.width] = 1
                adj[i + states - self.width, i] = 1

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran

        # always start at middle-ish state
        self.start_state = ((self.height // 2) * self.width) + self.width // 2

    def relation(self, s1, s2):
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == self.width or diff == -self.width * (self.height - 1):  # down
            rel_type = 'down'
        elif diff == -self.width or diff == self.width * (self.height - 1):  # up
            rel_type = 'up'
        elif diff == -1 or diff == (self.width - 1):  # left
            rel_type = 'left'
        elif diff == 1 or diff == -(self.width - 1):  # right
            rel_type = 'right'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = [i for i in range(self.par.o_size)]
        for i in range(self.n_states):
            states_vec[i] = np.random.choice(choices)
            if self.par.sample_observation_without_replacement:
                choices.remove(states_vec[i])

        self.states_mat = states_vec.astype(int)

    def walk(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        # consider rectangular environment. if move +1=up, -1=down, +length=right, -length=left.
        if self.height * self.width != len(self.adj):
            raise ValueError('incorrect height/width : height * width not equal to number of states')

        time_steps = self.walk_len
        time_steps_ = time_steps + self.n_states
        if self.par.behaviour_type in ['up,left,down,down,right,right,up,up']:
            position = np.zeros(time_steps_, dtype=np.int16)
            position[0] = int(self.start_state)
            velocities = [self.par.env.velocities[self.rels.index(x)] for x in self.par.behaviour_type.split(',')]
            tot_vels = len(velocities)
            width, height = position[0] % self.width, position[0] // self.width
            for i in range(time_steps_ - 1):
                vel = velocities[i % tot_vels]
                width = (width + vel[1]) % self.width
                height = (height - vel[0]) % self.height  # subtract here as down increases height
                position[i + 1] = int(height * self.width + width)
        else:
            raise ValueError('Behaviour type not allowed')

        # make rest of time-series
        reward = np.zeros(time_steps_, dtype=np.int16)
        action = np.zeros(time_steps_, dtype=np.int16)
        exploration = np.ones(time_steps_, dtype=np.int16)
        steps_between_rewards = np.zeros(time_steps_, dtype=np.int16)
        phase_velocity = np.zeros(time_steps_, dtype=np.float32)
        velocity = np.zeros((time_steps_, self.par.env.dim_space), dtype=np.float32)
        observation = np.zeros(time_steps_, dtype=np.int16)
        goal_position = np.zeros(time_steps_, dtype=np.int16)
        goal_observation = np.zeros(time_steps_, dtype=np.int16)

        observation[0] = self.states_mat[position[0]]
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        action[0] = 0

        visited_positions = [position[0]]
        for i in range(time_steps - 1):
            if position[i + 1] in visited_positions:
                exploration[i + 1] = 0
            else:
                visited_positions.append(position[i + 1])

            relation_taken, rel_type = self.relation(position[i], position[i + 1])
            action[i + 1] = relation_taken
            velocity[i + 1, :] = self.par.env.velocities[relation_taken]
            # stay still is just a set of zeros

            observation[i + 1] = self.states_mat[position[i + 1]]

        walk_data = {'position': position[:time_steps],
                     'action': action[:time_steps],
                     'reward': reward[:time_steps],
                     'exploration': exploration[:time_steps],
                     'steps_between_rewards': steps_between_rewards[:time_steps],
                     'phase_velocity': phase_velocity[:time_steps],
                     'phase': np.cumsum(phase_velocity) % 1.0,
                     'velocity': velocity[:time_steps],
                     'observation': observation[:time_steps],
                     'goal_position': goal_position[:time_steps],
                     'goal_observation': goal_observation[:time_steps],
                     'target_o': observation[:time_steps],
                     'chunk_action': np.zeros(time_steps, dtype=np.int16),
                     }

        return walk_data

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xs = xs.flatten() - (self.width - 1) / 2
        ys = - (ys.flatten() - (self.height - 1) / 2)

        if cells is not None:
            cell_prepared = cp.deepcopy(cells).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class RectangleChunk(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width, height)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states(width, height):
        return width * height

    def world(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        states = int(self.width * self.height)
        adj = np.zeros((states, states))

        for i in range(states):
            # stay still
            if self.par.env.stay_still:
                adj[i, i] = 1
            # up - down
            if i + self.width < states:
                adj[i, i + self.width] = 1
                adj[i + self.width, i] = 1
                # left - right
            if np.mod(i, self.width) != 0:
                adj[i, i - 1] = 1
                adj[i - 1, i] = 1

            if self.par.torus and np.mod(i, self.width) == 0:
                adj[i, i + self.width - 1] = 1
                adj[i + self.width - 1, i] = 1

            if self.par.torus and int(i / self.width) == 0:
                adj[i, i + states - self.width] = 1
                adj[i + states - self.width, i] = 1

        tran = np.zeros((states, states))
        for i in range(states):
            if sum(adj[i]) > 0:
                tran[i] = adj[i] / sum(adj[i])

        self.adj, self.tran = adj, tran

        # always start at middle-ish state
        self.start_state = ((self.height // 2) * self.width) + self.width // 2

    def relation(self, s1, s2):
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == self.width or diff == -self.width * (self.height - 1):  # down
            rel_type = 'down'
        elif diff == -self.width or diff == self.width * (self.height - 1):  # up
            rel_type = 'up'
        elif diff == -1 or diff == (self.width - 1):  # left
            rel_type = 'left'
        elif diff == 1 or diff == -(self.width - 1):  # right
            rel_type = 'right'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = [i for i in range(self.par.o_size)]
        for i in range(self.n_states):
            states_vec[i] = np.random.choice(choices)
            if self.par.sample_observation_without_replacement:
                choices.remove(states_vec[i])

        self.states_mat = states_vec.astype(int)

    def walk(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        # consider rectangular environment. if move +1=up, -1=down, +length=right, -length=left.
        if self.height * self.width != len(self.adj):
            raise ValueError('incorrect height/width : height * width not equal to number of states')

        time_steps = self.walk_len
        time_steps_ = time_steps + self.n_states

        position = np.zeros(time_steps_, dtype=np.int16)
        position[0] = int(self.start_state)
        width, height = position[0] % self.width, position[0] // self.width
        chunk_action = np.zeros(time_steps_, dtype=np.int16)  # dim space = num_chunks + 1
        chunk_vel = []
        for i in range(time_steps_ - 1):
            if len(chunk_vel) == 0:
                chunk_chosen = np.random.randint(len(self.par.chunks))
                chunk_vel = [self.par.env.velocities[self.rels.index(x)] for x in
                             self.par.chunks[chunk_chosen].split(',')]
                chunk_action[i + 1] = chunk_chosen + 1
            vel = chunk_vel.pop()
            width = (width + vel[1]) % self.width
            height = (height - vel[0]) % self.height  # subtract here as down increases height
            position[i + 1] = int(height * self.width + width)

        # make rest of time-series
        reward = np.zeros(time_steps_, dtype=np.int16)
        action = np.zeros(time_steps_, dtype=np.int16)
        exploration = np.ones(time_steps_, dtype=np.int16)
        steps_between_rewards = np.zeros(time_steps_, dtype=np.int16)
        phase_velocity = np.zeros(time_steps_, dtype=np.float32)
        velocity = np.zeros((time_steps_, self.par.env.dim_space), dtype=np.float32)
        observation = np.zeros(time_steps_, dtype=np.int16)
        goal_position = np.zeros(time_steps_, dtype=np.int16)
        goal_observation = np.zeros(time_steps_, dtype=np.int16)

        observation[0] = self.states_mat[position[0]]
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        action[0] = 0

        visited_positions = [position[0]]
        for i in range(time_steps - 1):
            if position[i + 1] in visited_positions:
                exploration[i + 1] = 0
            else:
                visited_positions.append(position[i + 1])

            relation_taken, rel_type = self.relation(position[i], position[i + 1])
            action[i + 1] = relation_taken
            velocity[i + 1, :] = self.par.env.velocities[relation_taken]
            # stay still is just a set of zeros

            observation[i + 1] = self.states_mat[position[i + 1]]

        walk_data = {'position': position[:time_steps],
                     'action': action[:time_steps],
                     'reward': reward[:time_steps],
                     'exploration': exploration[:time_steps],
                     'steps_between_rewards': steps_between_rewards[:time_steps],
                     'phase_velocity': phase_velocity[:time_steps],
                     'phase': np.cumsum(phase_velocity) % 1.0,
                     'velocity': velocity[:time_steps],
                     'observation': observation[:time_steps],
                     'goal_position': goal_position[:time_steps],
                     'goal_observation': goal_observation[:time_steps],
                     'target_o': observation[:time_steps],
                     'chunk_action': chunk_action[:time_steps],
                     }

        return walk_data

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xs = xs.flatten() - (self.width - 1) / 2
        ys = - (ys.flatten() - (self.height - 1) / 2)

        if cells is not None:
            cell_prepared = cp.deepcopy(cells).flatten()

            return xs, ys, cell_prepared
        else:
            return xs, ys


class Loop(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states_max(width):
        return width

    @staticmethod
    def get_n_states(width):
        return width

    def world(self):
        self.adj = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            # stay still
            if self.par.env.stay_still:
                self.adj[i, i] = 1
            # progress
            self.adj[i, np.mod(i + 1, self.n_states)] = 1

        self.tran = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if sum(self.adj[i]) > 0:
                self.tran[i] = self.adj[i] / sum(self.adj[i])

        # choose reward locations
        self.observation_locations = []
        for i in range(self.n_states):
            self.observation_locations.append(i)
        self.observation_locations.sort()  # just make this ordered for loop

        self.start_state = self.observation_locations[0]

    def relation(self, s1, s2):
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == 1 or diff == -(self.n_states - 1):
            rel_type = 'forward_1'
        elif diff == -1 or diff == (self.n_states - 1):
            rel_type = 'backward_1'
        elif diff == 2 or diff == -(self.n_states - 2):
            rel_type = 'forward_2'
        elif diff == -2 or diff == (self.n_states - 2):
            rel_type = 'backward_2'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = [i for i in range(self.par.o_size)]

        for i in self.observation_locations:
            states_vec[i] = np.random.choice(choices)
            if self.par.sample_observation_without_replacement:
                choices.remove(states_vec[i])

        self.states_mat = states_vec.astype(int)

    # @profile
    def walk(self):
        time_steps = self.walk_len
        time_steps_ = time_steps + self.n_states
        if self.par.behaviour_type in ['repeat', 'reverse', 'reverse_hierarchcial', 'skip_1']:
            # collect single loop
            position_snippet = [int(self.start_state)]  # np.zeros(self.n_states, dtype=np.int16)
            # position_snippet[0] = int(self.start_state)
            for i in range(self.n_states - 1):
                available = np.where(self.tran[int(position_snippet[i]), :] > 0)[0].astype(int)
                p = self.tran[int(position_snippet[i]), available]  # choose next position from actual allowed positions
                new_poss_pos = np.random.choice(available, p=p)

                if self.adj[position_snippet[i], new_poss_pos] == 1:
                    # position_snippet[i + 1] = new_poss_pos
                    position_snippet.append(new_poss_pos)
                else:
                    # position_snippet[i + 1] = int(cp.deepcopy(position_snippet[i]))
                    position_snippet.append(int(cp.deepcopy(position_snippet[i])))

            # now enact behaviour (generating positions!)
            mult = int(np.ceil(time_steps / self.n_states))
            pos_ = []
            for i in range(mult + 1):
                if self.par.behaviour_type == 'repeat':
                    pos_.extend(position_snippet)
                elif self.par.behaviour_type == 'reverse':
                    pos_.extend(position_snippet) if i % (2 * self.par.loops_before_rev) < self.par.loops_before_rev \
                        else pos_.extend(position_snippet[::-1])
                elif self.par.behaviour_type == 'reverse_2':
                    pos_.extend(position_snippet) if i % (2 * self.par.loops_before_rev) < self.par.loops_before_rev \
                        else pos_.extend(np.roll(position_snippet[::-1], -1).tolist())
                elif self.par.behaviour_type == 'skip_1':
                    pos_.extend(position_snippet) if i == 0 else pos_.extend(np.roll(pos_[-1], -1).tolist())
                elif self.par.behaviour_type == 'reverse_hierarchcial':
                    len_p = len(pos_)
                    if len_p == 0:
                        pos_ = position_snippet[:]
                    elif len_p == 1:
                        pos_.extend(position_snippet[::-1])
                    else:
                        pos_ += pos_[int(len_p / 2):]
                        pos_ += pos_[:int(len_p / 2)]
                elif self.par.behaviour_type == 'reverse_hierarchcial_2':
                    len_p = len(pos_)
                    if len_p == 0:
                        pos_ = position_snippet[:]
                    elif len_p == 1:
                        pos_.extend(np.roll(position_snippet[::-1], -1).tolist())
                    else:
                        pos_ += pos_[int(len_p / 2):]
                        pos_ += pos_[:int(len_p / 2)]
                else:
                    raise ValueError('Have not implemented that behaviour type')
                if len(pos_) > (mult + 1) * self.n_states:
                    break
            # position = np.concatenate(pos_)[:time_steps_]
            position = np.array(pos_)[:time_steps_]
        elif self.par.behaviour_type in ['1,2,0,-1', '1,-1,2,0', '2,1,0,-1,0,1', '2,1,0,-1', '1,1,-1,1,1,0',
                                         '1,1,-1,1,1,0,1,0,-1,0,1']:
            position = np.zeros(time_steps_, dtype=np.int16)
            position[0] = int(self.start_state)
            velocities = [int(x) for x in self.par.behaviour_type.split(',')]
            tot_vels = len(velocities)
            for i in range(time_steps_ - 1):
                vel = velocities[i % tot_vels]
                position[i + 1] = (position[i] + vel) % self.n_states
            # pos_ = [position_snippet]
            # while len(np.concatenate(pos_)) <= (mult + 1) * self.n_states:
            #    pos = pos_[-1][-1]
            #    aa = []
            #    for i in [int(a) for a in self.par.behaviour_type.split(',')]:
            #        pos = (pos + i) % self.n_states
            #        aa.append(pos)
            #    pos_.append(aa)
        elif self.par.behaviour_type in ['random']:
            position = np.zeros(time_steps_, dtype=np.int16)
            position[0] = int(self.start_state)
            vel = 0
            for i in range(time_steps_ - 1):
                if (np.random.rand() < self.par.env.repeat_vel_prob) and vel != 0:
                    # don't repeat if vel=0
                    pass
                else:
                    vel = np.random.choice([-1, 0, 1])
                position[i + 1] = (position[i] + vel) % self.n_states
            # pos_ = [[int(self.start_state)]]
            # while len(np.concatenate(pos_)) <= (mult + 1) * self.n_states:
            #    pos = pos_[-1][-1]
            #    pos = (pos + np.random.choice([-1, 0, 1])) % self.n_states
            #    pos_.append([pos])
        else:
            raise ValueError('Unallowed behaviour type')

        # make rest of time-series
        reward = np.zeros(time_steps_, dtype=np.int16)
        action = np.zeros(time_steps_, dtype=np.int16)
        exploration = np.ones(time_steps_, dtype=np.int16)
        steps_between_rewards = np.zeros(time_steps_, dtype=np.int16)
        phase_velocity = np.zeros(time_steps_, dtype=np.float32)
        velocity = np.zeros((time_steps_, self.par.env.dim_space), dtype=np.float32)
        observation = np.zeros(time_steps_, dtype=np.int16)
        goal_position = np.zeros(time_steps_, dtype=np.int16)
        goal_observation = np.zeros(time_steps_, dtype=np.int16)

        observation[0] = self.states_mat[position[0]]
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        action[0] = 0

        # detect if at an observation location
        if position[0] in self.observation_locations:
            reward[0] = 1
        steps_between_reward = 1

        visited_positions = [position[0]]
        for i in range(time_steps_ - 1):

            if position[i + 1] in visited_positions:
                exploration[i + 1] = 0
            else:
                visited_positions.append(position[i + 1])

            relation_taken, rel_type = self.relation(position[i], position[i + 1])
            action[i + 1] = relation_taken
            velocity[i + 1, :] = self.par.env.velocities[relation_taken]

            # stay still is just a set of zeros
            observation[i + 1] = self.states_mat[position[i + 1]]

            # detect if at a rewarded location
            if position[i + 1] in self.observation_locations:
                reward[i + 1] = 1
                # update steps between this reward and last
                for j in range(steps_between_reward):
                    steps_between_rewards[i + 1 - j] = steps_between_reward
                steps_between_reward = 0
            steps_between_reward += 1
        phase_velocity[1:time_steps] = 1. / steps_between_rewards[1:time_steps]

        walk_data = {'position': position[:time_steps],
                     'action': action[:time_steps],
                     'reward': reward[:time_steps],
                     'exploration': exploration[:time_steps],
                     'steps_between_rewards': steps_between_rewards[:time_steps],
                     'phase_velocity': phase_velocity[:time_steps],
                     'phase': np.cumsum(phase_velocity) % 1.0,
                     'velocity': velocity[:time_steps],
                     'observation': observation[:time_steps],
                     'goal_position': goal_position[:time_steps],
                     'goal_observation': goal_observation[:time_steps],
                     'target_o': observation[:time_steps],
                     'chunk_action': np.zeros(time_steps, dtype=np.int16),
                     }

        return walk_data

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        raise ValueError('NEED TO ADD THIS FUNCTION FOR LOOP')


class LoopChunk(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states_max(width):
        return width

    @staticmethod
    def get_n_states(width):
        return width

    def world(self):
        self.adj = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            # stay still
            if self.par.env.stay_still:
                self.adj[i, i] = 1
            # progress
            self.adj[i, np.mod(i + 1, self.n_states)] = 1

        self.tran = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if sum(self.adj[i]) > 0:
                self.tran[i] = self.adj[i] / sum(self.adj[i])

        # choose reward locations
        self.observation_locations = []
        for i in range(self.n_states):
            self.observation_locations.append(i)
        self.observation_locations.sort()  # just make this ordered for loop

        self.start_state = self.observation_locations[0]

    def relation(self, s1, s2):
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == 1 or diff == -(self.n_states - 1):
            rel_type = 'forward_1'
        elif diff == -1 or diff == (self.n_states - 1):
            rel_type = 'backward_1'
        elif diff == 2 or diff == -(self.n_states - 2):
            rel_type = 'forward_2'
        elif diff == -2 or diff == (self.n_states - 2):
            rel_type = 'backward_2'
        elif diff == 0:
            rel_type = 'stay still'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = [i for i in range(self.par.o_size)]

        for i in self.observation_locations:
            states_vec[i] = np.random.choice(choices)
            if self.par.sample_observation_without_replacement:
                choices.remove(states_vec[i])

        self.states_mat = states_vec.astype(int)

    # @profile
    def walk(self):
        time_steps = self.walk_len
        time_steps_ = time_steps + self.n_states

        position = np.zeros(time_steps_, dtype=np.int16)
        position[0] = int(self.start_state)
        chunk_action = np.zeros(time_steps_, dtype=np.int16)  # dim space = num_chunks + 1
        chunk_vel = []
        av_vel = 0
        for i in range(time_steps_ - 1):
            if len(chunk_vel) == 0:
                if (np.random.rand() < self.par.env.repeat_vel_prob) and i > 0 and av_vel != 0:
                    pass
                else:
                    chunk_chosen = np.random.randint(len(self.par.chunks))
                chunk_vel = [int(x) for x in self.par.chunks[chunk_chosen].split(',')]
                av_vel = np.mean(chunk_vel)
                chunk_action[i + 1] = chunk_chosen + 1
            vel = chunk_vel.pop()
            position[i + 1] = (position[i] + vel) % self.n_states

        # make rest of time-series
        reward = np.zeros(time_steps_, dtype=np.int16)
        action = np.zeros(time_steps_, dtype=np.int16)
        exploration = np.ones(time_steps_, dtype=np.int16)
        steps_between_rewards = np.zeros(time_steps_, dtype=np.int16)
        phase_velocity = np.zeros(time_steps_, dtype=np.float32)
        velocity = np.zeros((time_steps_, self.par.env.dim_space), dtype=np.float32)
        observation = np.zeros(time_steps_, dtype=np.int16)
        goal_position = np.zeros(time_steps_, dtype=np.int16)
        goal_observation = np.zeros(time_steps_, dtype=np.int16)

        observation[0] = self.states_mat[position[0]]
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        action[0] = 0

        # detect if at an observation location
        if position[0] in self.observation_locations:
            reward[0] = 1
        steps_between_reward = 1

        visited_positions = [position[0]]
        for i in range(time_steps_ - 1):

            if position[i + 1] in visited_positions:
                exploration[i + 1] = 0
            else:
                visited_positions.append(position[i + 1])

            relation_taken, rel_type = self.relation(position[i], position[i + 1])
            action[i + 1] = relation_taken
            velocity[i + 1, :] = self.par.env.velocities[relation_taken]

            # stay still is just a set of zeros
            observation[i + 1] = self.states_mat[position[i + 1]]

            # detect if at a rewarded location
            if position[i + 1] in self.observation_locations:
                reward[i + 1] = 1
                # update steps between this reward and last
                for j in range(steps_between_reward):
                    steps_between_rewards[i + 1 - j] = steps_between_reward
                steps_between_reward = 0
            steps_between_reward += 1
        phase_velocity[1:time_steps] = 1. / steps_between_rewards[1:time_steps]

        walk_data = {'position': position[:time_steps],
                     'action': action[:time_steps],
                     'reward': reward[:time_steps],
                     'exploration': exploration[:time_steps],
                     'steps_between_rewards': steps_between_rewards[:time_steps],
                     'phase_velocity': phase_velocity[:time_steps],
                     'phase': np.cumsum(phase_velocity) % 1.0,
                     'velocity': velocity[:time_steps],
                     'observation': observation[:time_steps],
                     'goal_position': goal_position[:time_steps],
                     'goal_observation': goal_observation[:time_steps],
                     'target_o': observation[:time_steps],
                     'chunk_action': chunk_action[:time_steps],
                     }

        return walk_data

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        raise ValueError('NEED TO ADD THIS FUNCTION FOR LOOP')


class LoopDelay(Environment):

    def __init__(self, params, width, height):
        self.delay = np.random.randint(params.env.delay_min, params.env.delay_max + 1, width)
        try:
            if params.env.same_delays:
                self.delay = self.delay[0] * np.ones_like(self.delay)
        except (KeyError, AttributeError) as e:
            pass
        self.n_states = self.get_n_states(width)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states_max(width, max_delay):
        return width + width * max_delay

    def get_n_states(self, width):
        return width + sum(self.delay)

    def world(self):
        self.adj = np.zeros((self.n_states, self.n_states))
        self.tran = np.zeros((self.n_states, self.n_states))

        # choose reward locations
        self.observation_locations = []
        for i, delay in enumerate(np.append(0, np.cumsum(self.delay))[:-1]):
            self.observation_locations.append(i + delay)  # first observation always at location 0
        self.observation_locations.sort()  # just make this ordered for loop
        self.start_state = self.observation_locations[0]

    def relation(self, s1, s2):
        if s2 in self.observation_locations:
            rel_type = 'forward_1'
        else:
            rel_type = 'stay still'
        rel_index = self.rels.index(rel_type)
        return rel_index, rel_type

    def state_data(self):
        states_vec = (self.par.o_size - 1) * np.ones(self.n_states)  # o_sie -1 as that's delay observation
        choices = [i for i in range(self.par.o_size - 1)]

        for i in self.observation_locations:
            states_vec[i] = np.random.choice(choices)
            if self.par.sample_observation_without_replacement:
                choices.remove(states_vec[i])

        self.states_mat = states_vec.astype(int)

    # @profile
    def walk(self):
        time_steps = self.walk_len

        # collect single loop
        position_snippet = np.zeros(self.n_states, dtype=np.int16)
        position_snippet[0] = int(self.start_state)
        for i in range(self.n_states - 1):
            position_snippet[i + 1] = position_snippet[i] + 1

        # tile positions
        mult = int(np.ceil(time_steps / self.n_states)) + 1
        position = np.concatenate([position_snippet] * mult)

        # make rest of time-series
        time_steps_ = len(position)
        reward = np.zeros(time_steps_, dtype=np.int16)
        action = np.zeros(time_steps_, dtype=np.int16)
        exploration = np.ones(time_steps_, dtype=np.int16)
        steps_between_rewards = np.zeros(time_steps_, dtype=np.int16)
        phase_velocity = np.zeros(time_steps_, dtype=np.float32)
        velocity = np.zeros((time_steps_, self.par.env.dim_space), dtype=np.float32)
        observation = np.zeros(time_steps_, dtype=np.int16)
        goal_position = np.zeros(time_steps_, dtype=np.int16)
        goal_observation = np.zeros(time_steps_, dtype=np.int16)

        observation[0] = self.states_mat[position[0]]
        # choose random action to have gotten to start-state - doesn't get used as g_prior is for first state
        action[0] = 0
        goal_index = 0

        # detect if at an observation location
        if position[0] in self.observation_locations:
            goal_index = (goal_index + 1) % self.width
            reward[0] = 1
        goal_position[0] = self.observation_locations[goal_index]
        goal_observation[0] = self.states_mat[goal_position[0]]
        steps_between_reward = 1

        visited_positions = [position[0]]
        for i in range(time_steps_ - 1):

            if position[i + 1] in visited_positions:
                exploration[i + 1] = 0
            else:
                visited_positions.append(position[i + 1])

            relation_taken, rel_type = self.relation(position[i], position[i + 1])
            action[i + 1] = relation_taken
            velocity[i + 1, :] = self.par.env.velocities[relation_taken]

            # stay still is just a set of zeros
            observation[i + 1] = self.states_mat[position[i + 1]]

            # detect if at a rewarded location
            if position[i + 1] in self.observation_locations:
                goal_index = (goal_index + 1) % self.width
                reward[i + 1] = 1
                # update steps between this reward and last
                for j in range(steps_between_reward):
                    steps_between_rewards[i + 1 - j] = steps_between_reward
                steps_between_reward = 0
            goal_position[i + 1] = self.observation_locations[goal_index]
            goal_observation[i + 1] = self.states_mat[goal_position[i + 1]]
            steps_between_reward += 1
        phase_velocity[1:time_steps] = 1. / steps_between_rewards[1:time_steps]

        exploration[self.n_states] = 1  # this is because it's impossible ot know 'length' of loop until seen orig again

        walk_data = {'position': position[:time_steps],
                     'action': action[:time_steps],
                     'reward': reward[:time_steps],
                     'exploration': exploration[:time_steps],
                     'steps_between_rewards': steps_between_rewards[:time_steps],
                     'phase_velocity': phase_velocity[:time_steps],
                     'phase': np.cumsum(phase_velocity)[:time_steps] % 1.0,
                     'velocity': velocity[:time_steps],
                     'observation': observation[:time_steps],
                     'goal_position': goal_position[:time_steps],
                     'goal_observation': goal_observation[:time_steps],
                     'target_o': observation[:time_steps],
                     'chunk_action': np.zeros(time_steps, dtype=np.int16),
                     }

        return walk_data

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        raise ValueError('NEED TO ADD THIS FUNCTION FOR LOOP')


class Line(Loop):
    def __init__(self, params, width, height):
        super().__init__(params, width, height)
        self.delay = np.random.randint(params.env.delay_min, params.env.delay_max + 1, width)
        self.n_states = self.get_n_states(width)

    def world(self):
        self.adj = np.zeros((self.n_states, self.n_states))

        for i in range(self.n_states):
            # stay still
            if self.par.env.stay_still:
                self.adj[i, i] = 1
            # progress
            if i < self.n_states:
                self.adj[i, i + 1] = 1
                self.adj[i + 1, i] = 1

        self.tran = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if sum(self.adj[i]) > 0:
                self.tran[i] = self.adj[i] / sum(self.adj[i])

        # choose reward locations
        self.reward_locations = []
        for i, delay in enumerate(np.append(0, np.cumsum(self.delay))[:-1]):
            self.reward_locations.append(i + delay)
        self.reward_locations.sort()  # just make this ordered for loop

        self.start_state = self.reward_locations[0]


class NBack(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states(width):
        return width

    def world(self):
        self.adj = np.zeros((self.n_states, self.n_states))
        self.tran = np.zeros((self.n_states, self.n_states))

    def relation(self, s1, s2):
        return 0, 'forward'

    def state_data(self):
        self.states_mat = np.zeros(self.n_states)

    # @profile
    def walk(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        time_steps = self.walk_len

        observation = np.random.randint(low=0, high=self.par.o_size, size=time_steps)
        exploration = np.zeros(time_steps, dtype=np.int16)
        exploration[:self.width] = 1
        phase_velocity = np.ones(time_steps, dtype=np.float32)

        walk_data = {'position': np.tile(np.arange(self.width), int(np.ceil(time_steps / self.width)))[:time_steps],
                     'action': np.zeros(time_steps, dtype=np.int16),
                     'reward': np.ones(time_steps, dtype=np.int16),
                     'exploration': exploration,
                     'steps_between_rewards': np.ones(time_steps, dtype=np.int16),
                     'phase_velocity': phase_velocity,
                     'velocity': np.ones((time_steps, self.par.env.dim_space), dtype=np.float32),
                     'observation': observation,
                     'phase': np.cumsum(phase_velocity) % 1.0,
                     'goal_position': np.zeros(time_steps, dtype=np.int16),
                     'goal_observation': np.zeros(time_steps, dtype=np.int16),
                     'target_o': np.roll(observation, self.width),
                     'chunk_action': np.zeros(time_steps, dtype=np.int16),
                     }

        return walk_data

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        raise ValueError('NEED TO ADD THIS FUNCTION FOR LOOP')


class Panichello2021(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width)
        super().__init__(params, width, height)

    @staticmethod
    def get_n_states(width):
        return width

    def world(self):
        self.adj = np.zeros((self.n_states, self.n_states))
        self.tran = np.zeros((self.n_states, self.n_states))

    def relation(self, s1, s2):
        raise ValueError('not implemented')

    def state_data(self):
        self.states_mat = np.zeros(self.n_states)

    # @profile
    def walk(self):
        """
        #state number counts accross then down
        a = np.asarray(range(25))
        print(a)
        print(np.reshape(a,(5,5)))
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24]
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]
         [20 21 22 23 24]]
        """
        time_steps = self.walk_len

        # here n_observation should be n_poss + 2 + 1
        # where 2 is two cues, and 1 is extra is 'no observation at selection time'

        observation = np.random.randint(low=0, high=self.par.o_size - self.width - 1, size=time_steps)
        # choose whether a cue_1 or a cue_2 trial.
        cue_index = np.random.randint(low=0, high=self.width)
        observation[-2] = self.par.o_size - self.width - 1 + cue_index  # cue observation
        observation[-1] = self.par.o_size - 1  # dummy observation
        target_o = np.zeros(time_steps, dtype=np.int16)
        target_o[-1] = observation[cue_index]

        exploration = np.zeros(time_steps, dtype=np.int16)
        exploration[:-1] = 1

        position = np.arange(self.walk_len)
        # below: as observation is cue index here... so don't move
        # (network could choose to do lots of other things here though... and compensate it next step)
        position[-2] = position[-3]
        position[-1] = position[cue_index]

        rel_type = ['observation' for _ in range(self.width)] + ['stay_still'] + ['cue_' + str(cue_index)]
        action = np.array([self.rels.index(rt) for rt in rel_type], dtype=np.int16)
        velocity = np.zeros((time_steps, self.par.env.dim_space), dtype=np.float32)
        for i, a in enumerate(action):
            velocity[i, :] = self.par.env.velocities[a]
        phase_velocity = np.ones(time_steps, dtype=np.float32)

        walk_data = {'position': position,
                     'action': action,
                     'reward': np.ones(time_steps, dtype=np.int16),
                     'exploration': exploration,
                     'steps_between_rewards': np.ones(time_steps, dtype=np.int16),
                     'phase_velocity': phase_velocity,
                     'velocity': velocity,
                     'observation': observation,
                     'phase': np.cumsum(phase_velocity) % 1.0,
                     'goal_position': np.zeros(time_steps, dtype=np.int16),
                     'goal_observation': np.zeros(time_steps, dtype=np.int16),
                     'target_o': target_o,
                     'chunk_action': np.zeros(time_steps, dtype=np.int16),
                     }

        return walk_data

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):
        raise ValueError('NEED TO ADD THIS FUNCTION FOR LOOP')


class Xie2022(Environment):

    def __init__(self, params, width, height):
        self.n_states = self.get_n_states(width)
        super().__init__(params, width, height)

    def get_n_states(self, width):
        return width

    def world(self):
        self.adj = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            # progress
            self.adj[i, np.mod(i + 1, self.n_states)] = 1
        self.tran = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if sum(self.adj[i]) > 0:
                self.tran[i] = self.adj[i] / sum(self.adj[i])

        self.start_state = 0

    def relation(self, s1, s2):
        # consider square environment. if move +1=up, -1=down, +length=right, -length=left.
        diff = s2 - s1
        if diff == 1 or diff == -(self.n_states - 1):
            rel_type = 'forward'
        else:
            raise ValueError('impossible action')

        rel_index = self.rels.index(rel_type)

        return rel_index, rel_type

    def state_data(self):
        states_vec = np.zeros(self.n_states)
        choices = [i for i in range(self.par.o_size - 1)]

        for i in range(self.n_states):
            states_vec[i] = np.random.choice(choices)
            if self.par.sample_observation_without_replacement:
                choices.remove(states_vec[i])

        self.states_mat = states_vec.astype(int)

    # @profile
    def walk(self):
        time_steps = self.walk_len
        loop_index = int(time_steps / 2)
        observation = np.ones(time_steps, dtype=np.int16) * (self.par.o_size - 1)
        observation[:loop_index] = self.states_mat
        target_o = np.ones(time_steps, dtype=np.int16) * (self.par.o_size - 1)  # dummy target
        target_o[loop_index:] = self.states_mat

        exploration = np.zeros(time_steps, dtype=np.int16)
        exploration[:loop_index] = 1

        phase_velocity = np.ones(time_steps, dtype=np.float32)

        walk_data = {'position': np.tile(np.arange(loop_index), 2).astype(int),
                     'action': np.zeros(time_steps, dtype=np.int16),
                     'reward': np.ones(time_steps, dtype=np.int16),
                     'exploration': exploration,
                     'steps_between_rewards': np.ones(time_steps, dtype=np.int16),
                     'phase_velocity': phase_velocity,
                     'velocity': np.ones((time_steps, self.par.env.dim_space), dtype=np.float32),
                     'observation': observation,
                     'phase': np.cumsum(phase_velocity) % 1.0,
                     'goal_position': np.zeros(time_steps, dtype=np.int16),
                     'goal_observation': np.zeros(time_steps, dtype=np.int16),
                     'target_o': target_o,
                     'chunk_action': np.zeros(time_steps, dtype=np.int16),
                     }

        return walk_data

    def get_node_positions(self, cells=None, _plot_specs=None, _mask=None):

        raise ValueError('NEED TO ADD THIS FUNCTION FOR LOOP')


def get_new_data_diff_envs(position, pars, envs_class):
    b_s = int(pars.batch_size)
    n_walk = position.shape[-1]  # pars.seq_len
    o_size = pars.o_size

    data = np.zeros((b_s, o_size, n_walk))
    for batch in range(b_s):
        data[batch] = sample_data(position[batch, :], envs_class[batch].states_mat, o_size)

    return data


def sample_data(position, states_mat, o_size):
    # makes one-hot encoding of sensory at each time-step
    time_steps = np.shape(position)[0]
    sense_data = np.zeros((o_size, time_steps))
    for i, pos in enumerate(position):
        ind = int(pos)
        sense_data[states_mat[ind], i] = 1
    return sense_data


def square2hex(a):
    # length must be odd
    n_states = len(a)
    length = int(np.sqrt(len(a)))
    hex_length = (length + 1) / 2

    middle = int((n_states - 1) / 2)
    init = np.zeros(n_states)
    init[middle] = 1

    n_hops = int(hex_length - 1)
    jumps = [init]
    for i in range(n_hops):
        jumps.append(np.dot(a, jumps[i]))

    jumps_add = np.sum(jumps, 0)

    a_new = cp.deepcopy(a)
    for i, val in enumerate(list(jumps_add)):
        if val == 0:
            a_new[i, :] = 0
            a_new[:, i] = 0

    return a_new


def in_hexagon(x, y, width):
    # x, y, are centered about middle
    return np.abs(y) / np.sqrt(3) <= np.minimum(width / 4, width / 2 - np.abs(x))
