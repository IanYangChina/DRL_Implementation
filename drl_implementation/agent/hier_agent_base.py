import os
import torch as T
import numpy as np
import json
from collections import namedtuple
from .utils.plot import smoothed_plot
from .utils.replay_buffer import *
from .utils.normalizer import Normalizer
t = namedtuple("transition", ('state', 'action', 'next_state', 'reward', 'done'))
t_goal = namedtuple("transition", ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done'))


def mkdir(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


class Agent(object):
    # a base class for any two-level hierarchical agent
    def __init__(self,
                 algo_params,
                 transition_tuple=None,
                 image_obs=False, action_type='continuous',
                 goal_conditioned=False, path=None, seed=-1):
        # torch device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        if 'cuda_device_id' in algo_params.keys():
            self.device = T.device("cuda:%i" % algo_params['cuda_device_id'])
        # path & seeding
        T.manual_seed(seed)
        T.cuda.manual_seed_all(seed)  # this has no effect if cuda is not available

        # create a random number generator and seed it
        self.rng = np.random.default_rng(seed=seed)
        assert path is not None, 'please specify a project path to save files'
        self.path = path
        # path to save neural network check point
        self.ckpt_path = os.path.join(path, 'ckpts')
        # path to save statistics
        self.data_path = os.path.join(path, 'data')
        # create directories if not exist
        mkdir([self.path, self.ckpt_path, self.data_path])

        # non-goal-conditioned args
        self.image_obs = image_obs
        self.action_type = action_type
        self.state_dim = algo_params['state_dim']
        self.action_dim = algo_params['action_dim']
        if self.action_type == 'continuous':
            self.action_max = algo_params['action_max']
            self.action_scaling = algo_params['action_scaling']

        # prioritised replay
        self.prioritised = algo_params['prioritised']
        # non-goal-conditioned replay buffer
        tr = transition_tuple
        if transition_tuple is None:
            tr = t
        if not self.prioritised:
            self.buffer = ReplayBuffer(algo_params['memory_capacity'], tr, seed=seed)
        else:
            self.buffer = PrioritisedReplayBuffer(algo_params['memory_capacity'], tr, rng=self.rng)

        # goal-conditioned args & buffers
        self.goal_conditioned = goal_conditioned
        if self.goal_conditioned:
            if self.image_obs:
                self.goal_shape = algo_params['goal_shape']
            else:
                self.goal_dim = algo_params['goal_dim']
            self.hindsight = algo_params['hindsight']
            if transition_tuple is None:
                tr = t_goal
            if not self.prioritised:
                self.buffer = HindsightReplayBuffer(algo_params['memory_capacity'], tr,
                                                    sampling_strategy=algo_params['her_sampling_strategy'],
                                                    sampled_goal_num=4,
                                                    terminate_on_achieve=algo_params['terminate_on_achieve'],
                                                    seed=seed)
            else:
                self.buffer = PrioritisedHindsightReplayBuffer(algo_params['memory_capacity'],
                                                               tr,
                                                               sampling_strategy=algo_params['her_sampling_strategy'],
                                                               sampled_goal_num=4,
                                                               terminate_on_achieve=algo_params['terminate_on_achieve'],
                                                               rng=self.rng)
        else:
            self.goal_dim = 0

        # common args
        self.observation_normalization = algo_params['observation_normalization']
        # if using image obs, normalizer returns inputs/255.
        # if not using obs normalization, the normalizer is just a scale multiplier, returns inputs*scale
        self.normalizer = Normalizer(self.state_dim+self.goal_dim,
                                     algo_params['init_input_means'], algo_params['init_input_vars'],
                                     image_obs=self.image_obs,
                                     activated=self.observation_normalization)
        self.actor_learning_rate = algo_params['actor_learning_rate']
        self.critic_learning_rate = algo_params['critic_learning_rate']
        self.update_interval = algo_params['update_interval']
        self.batch_size = algo_params['batch_size']
        self.optimizer_steps = algo_params['optimization_steps']
        self.gamma = algo_params['discount_factor']
        self.discard_time_limit = algo_params['discard_time_limit']
        self.tau = algo_params['tau']
        self.optim_step_count = 0
        self.env_step_count = 0

        # network dict is filled in each specific agent
        self.network_dict = {}
        self.network_keys_to_save = None

        # algorithm-specific statistics are defined in each agent sub-class
        self.statistic_dict = {
            # use lowercase characters
            'actor_loss': [],
            'critic_loss': [],
        }

    def run(self, render=False, test=False, load_network_ep=None, sleep=0):
        raise NotImplementedError()

    def _interact(self, render=False, test=False, sleep=0):
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
        if not self.image_obs:
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
        if not self.image_obs:
            np.save(os.path.join(self.data_path, 'input_means'), self.normalizer.history_mean)
            np.save(os.path.join(self.data_path, 'input_vars'), self.normalizer.history_var)
        json.dump(self.statistic_dict, open(os.path.join(self.data_path, 'statistics.json'), 'w'))
    
    def _plot_statistics(self, keys=None, x_labels=None, y_labels=None, window=5):
        if y_labels is None:
            y_labels = {}
        for key in list(self.statistic_dict.keys()):
            if key not in y_labels.keys():
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
            if key not in x_labels.keys():
                if ('loss' in key) or ('alpha' in key) or ('entropy' in key):
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
        else:
            for key in keys:
                smoothed_plot(os.path.join(self.path, key+'.png'), self.statistic_dict[key],
                              x_label=x_labels[key], y_label=y_labels[key], window=window)

