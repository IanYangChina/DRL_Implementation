import os
import torch as T
import numpy as np
import json
from collections import namedtuple
from plot import smoothed_plot
from agent.utils.replay_buffer import *
from agent.utils.normalizer import Normalizer
t = namedtuple("transition", ('state', 'action', 'next_state', 'reward', 'done'))
t_goal = namedtuple("transition", ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))


def mkdir(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


class Agent(object):
    def __init__(self, algo_params, transition_tuple=None, goal_conditioned=False, path=None, seed=-1):
        # path & seeding
        T.manual_seed(seed)
        self.rng = np.random.default_rng(seed=seed)
        assert path is not None, 'please specify a project path'
        self.path = path
        self.ckpt_path = os.path.join(path, 'ckpts')
        self.data_path = os.path.join(path, 'data')
        mkdir([self.path, self.ckpt_path, self.data_path])

        # torch device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # non-goal-conditioned args
        self.state_dim = algo_params['state_dim']
        self.action_dim = algo_params['action_dim']
        self.action_max = algo_params['action_max']
        self.prioritised = algo_params['prioritised']
        tr = transition_tuple
        if transition_tuple is None:
            tr = t
        if not self.prioritised:
            self.buffer = ReplayBuffer(algo_params['memory_capacity'], tr, seed=seed)
        else:
            self.buffer = PrioritisedReplayBuffer(algo_params['memory_capacity'], tr, rng=self.rng)

        # goal-conditioned args
        self.goal_conditioned = goal_conditioned
        if self.goal_conditioned:
            self.goal_dim = algo_params['goal_dim']
            self.hindsight = algo_params['hindsight']
            if transition_tuple is None:
                tr = t_goal
            if not self.prioritised:
                self.buffer = HindsightReplayBuffer(algo_params['memory_capacity'], tr, sampled_goal_num=4, seed=seed)
            else:
                self.buffer = PrioritisedHindsightReplayBuffer(algo_params['memory_capacity'], tr, rng=self.rng)
        else:
            self.goal_dim = 0

        # common args
        self.normalizer = Normalizer(self.state_dim+self.goal_dim,
                                     algo_params['init_input_means'], algo_params['init_input_vars'])
        self.learning_rate = algo_params['learning_rate']
        self.batch_size = algo_params['batch_size']
        self.optimizer_steps = algo_params['optimization_steps']
        self.gamma = algo_params['discount_factor']
        self.discard_time_limit = algo_params['discard_time_limit']
        self.tau = algo_params['tau']
        self.network_dict = {}
        self.network_keys_to_save = None
        self.statistic_dict = {
            # use lowercase characters
            'actor_loss': [],
            'critic_loss': [],
        }

    def run(self, render=False, test=False, load_network_ep=None):
        raise NotImplementedError()

    def _interact(self, render=False, test=False):
        raise NotImplementedError()

    def _select_action(self, obs, test=False):
        raise NotImplementedError()

    def _remember(self, *args, new_episode=False):
        if self.goal_conditioned:
            self.buffer.new_episode = new_episode
            self.buffer.store_experience(*args)
        else:
            self.buffer.store_experience(*args)

    def _learn(self, steps=None):
        raise NotImplementedError()

    def _soft_update(self, source, target, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def _save_network(self, keys=None, ep=None):
        if ep is None:
            ep = ''
        else:
            ep = '_ep'+str(ep)
        if keys is None:
            keys = self.network_keys_to_save
        assert keys is not None
        for key in keys:
            T.save(self.network_dict[key].state_dict(), self.ckpt_path+'/ckpt_'+key+ep+'.pt')

    def _load_network(self, keys=None, ep=None):
        self.normalizer.history_mean = np.load(os.path.join(self.data_path, 'input_means.npy'))
        self.normalizer.history_var = np.load(os.path.join(self.data_path, 'input_vars.npy'))
        if ep is None:
            ep = ''
        else:
            ep = '_ep'+str(ep)
        if keys is None:
            keys = self.network_keys_to_save
        assert keys is not None
        for key in keys:
            self.network_dict[key].load_state_dict(T.load(self.ckpt_path+'/ckpt_'+key+ep+'.pt'))

    def _save_statistics(self):
        np.save(os.path.join(self.data_path, 'input_means'), self.normalizer.history_mean)
        np.save(os.path.join(self.data_path, 'input_vars'), self.normalizer.history_var)
        json.dump(self.statistic_dict, open(os.path.join(self.data_path, 'statistics.json'), 'w'))
    
    def _plot_statistics(self, keys=None, x_labels=None, y_labels=None, window=5):
        if y_labels is None:
            y_labels = {}
            for key in list(self.statistic_dict.keys()):
                if 'loss' in key:
                    label = 'Loss'
                elif 'return' in key:
                    label = 'Return'
                elif 'success' in key:
                    label = 'Success'
                else:
                    label = key
                y_labels.update({key: label})
        
        if x_labels is None:
            x_labels = {}
            for key in list(self.statistic_dict.keys()):
                if 'loss' in key:
                    label = 'Optimization step'
                elif 'cycle' in key:
                    label = 'Cycle'
                elif 'epoch' in key:
                    label = 'Epoch'
                else:
                    label = 'Episode'
                x_labels.update({key: label})
        
        if keys is None:
            for key in list(self.statistic_dict.keys()):
                smoothed_plot(os.path.join(self.path, key+'.png'), self.statistic_dict[key],
                              x_label=x_labels[key], y_label=y_labels[key], window=window)
