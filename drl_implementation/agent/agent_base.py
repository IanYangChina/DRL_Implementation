import os
import logging as info_logging
import torch as T
import numpy as np
import json
import subprocess as sp
from torch.utils.tensorboard import SummaryWriter
from .utils.plot import smoothed_plot
from .utils.replay_buffer import make_buffer
from .utils.normalizer import Normalizer


def mkdir(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def reset_logging(logging_to_reset):
    loggers = [logging_to_reset.getLogger(name) for name in logging_to_reset.root.manager.loggerDict]
    loggers.append(logging_to_reset.getLogger())
    for logger in loggers:
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging_to_reset.NOTSET)
        logger.propagate = True


class Agent(object):
    def __init__(self,
                 algo_params, logging=None, create_logger=False,
                 transition_tuple=None,
                 non_flat_obs=False, action_type='continuous',
                 goal_conditioned=False, store_goal_ind=False, training_mode='episode_based',
                 path=None, log_dir_suffix=None, seed=-1):
        """
        Parameters
        ----------
        algo_params : dict
            a dictionary of parameters
        transition_tuple : collections.namedtuple
            a python namedtuple for storing, managing and sampling experiences, see .utils.replay_buffer
        non_flat_obs : bool
            whether the observations are 1D or nD
        action_type : str
            either 'discrete' or 'continuous'
        goal_conditioned : bool
            whether the agent uses a goal-conditioned policy
        training_mode : str
            either 'episode_based' or 'env_step_based'
        path : str
            a directory to save files
        seed : int
            a random seed
        """
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
        if log_dir_suffix is not None:
            comment = '-'+log_dir_suffix
        else:
            comment = ''
        self.create_logger = create_logger
        if self.create_logger:
            self.logger = SummaryWriter(log_dir=self.data_path, comment=comment)
        self.logging = logging
        if self.logging is None:
            reset_logging(info_logging)
            log_file_name = os.path.join(self.data_path, 'optimisation.log')
            if os.path.isfile(log_file_name):
                filemode = "a"
            else:
                filemode = "w"
            info_logging.basicConfig(level=info_logging.NOTSET, filemode=filemode,
                                filename=log_file_name,
                                format="%(asctime)s %(levelname)s %(message)s")
            self.logging = info_logging

        # non-goal-conditioned args
        self.non_flat_obs = non_flat_obs
        self.action_type = action_type
        if self.non_flat_obs:
            self.state_dim = 0
            self.state_shape = algo_params['state_shape']
        else:
            self.state_dim = algo_params['state_dim']
        if self.action_type == 'hybrid':
            self.discrete_action_dim = algo_params['discrete_action_dim']
            self.continuous_action_dim = algo_params['continuous_action_dim']
            self.continuous_action_max = algo_params['continuous_action_max']
            self.continuous_action_scaling = algo_params['continuous_action_scaling']
        else:
            self.action_dim = algo_params['action_dim']
        if self.action_type == 'continuous':
            self.action_max = algo_params['action_max']
            self.action_scaling = algo_params['action_scaling']

        # goal-conditioned args & buffers
        self.goal_conditioned = goal_conditioned
        # prioritised replay
        self.prioritised = algo_params['prioritised']

        if self.goal_conditioned:
            if self.non_flat_obs:
                self.goal_dim = 0
                self.goal_shape = algo_params['goal_shape']
            else:
                self.goal_dim = algo_params['goal_dim']
            self.hindsight = algo_params['hindsight']
            try:
                goal_distance_threshold = self.env.env.distance_threshold
            except:
                goal_distance_threshold = self.env.distance_threshold

            goal_conditioned_reward_func = None
            try:
                if self.env.env.goal_conditioned_reward_function is not None:
                    goal_conditioned_reward_func = self.env.env.goal_conditioned_reward_function
            except:
                if self.env.goal_conditioned_reward_function is not None:
                    goal_conditioned_reward_func = self.env.goal_conditioned_reward_function

            try:
                her_sample_strategy = algo_params['her_sampling_strategy']
            except:
                her_sample_strategy = 'future'

            try:
                num_sampled_goal = algo_params['num_sampled_goal']
            except:
                num_sampled_goal = 4

            self.buffer = make_buffer(mem_capacity=algo_params['memory_capacity'],
                                      transition_tuple=transition_tuple, prioritised=self.prioritised,
                                      seed=seed, rng=self.rng,
                                      goal_conditioned=True, keep_episode=self.hindsight,
                                      store_goal_ind=store_goal_ind,
                                      sampling_strategy=her_sample_strategy,
                                      num_sampled_goal=num_sampled_goal,
                                      terminal_on_achieved=algo_params['terminate_on_achieve'],
                                      goal_distance_threshold=goal_distance_threshold,
                                      goal_conditioned_reward_func=goal_conditioned_reward_func)
        else:
            self.goal_dim = 0
            self.buffer = make_buffer(mem_capacity=algo_params['memory_capacity'],
                                      transition_tuple=transition_tuple, prioritised=self.prioritised,
                                      seed=seed, rng=self.rng,
                                      goal_conditioned=False)

        # common args
        if not self.non_flat_obs:
            self.observation_normalization = algo_params['observation_normalization']
            # if not using obs normalization, the normalizer is just a scale multiplier, returns inputs*scale
            self.normalizer = Normalizer(self.state_dim+self.goal_dim,
                                         algo_params['init_input_means'], algo_params['init_input_vars'],
                                         activated=self.observation_normalization)
        try:
            self.update_interval = algo_params['update_interval']
        except:
            self.update_interval = 1
        self.actor_learning_rate = algo_params['actor_learning_rate']
        self.critic_learning_rate = algo_params['critic_learning_rate']
        self.batch_size = algo_params['batch_size']
        self.optimizer_steps = algo_params['optimization_steps']
        self.gamma = algo_params['discount_factor']
        self.discard_time_limit = algo_params['discard_time_limit']
        self.tau = algo_params['tau']
        self.optim_step_count = 0

        assert training_mode in ['episode_based', 'step_based']
        self.training_mode = training_mode
        self.total_env_step_count = 0
        self.total_env_episode_count = 0

        # network dict is filled in each specific agent
        self.network_dict = {}
        self.network_keys_to_save = None

        # algorithm-specific statistics are defined in each agent sub-class
        self.statistic_dict = {
            # use lowercase characters
            'actor_loss': [],
            'critic_loss': [],
        }
        self.get_gpu_memory = get_gpu_memory

    def run(self, render=False, test=False, load_network_ep=None, sleep=0):
        raise NotImplementedError

    def _interact(self, render=False, test=False, sleep=0):
        raise NotImplementedError

    def _select_action(self, obs, test=False):
        raise NotImplementedError

    def _learn(self, steps=None):
        raise NotImplementedError

    def _remember(self, *args, new_episode=False):
        if self.goal_conditioned:
            self.buffer.new_episode = new_episode
            self.buffer.store_experience(*args)
        else:
            self.buffer.store_experience(*args)

    def _soft_update(self, source, target, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def _save_network(self, keys=None, ep=None, step=None):
        if ep is None:
            ep = ''
        else:
            ep = '_ep'+str(ep)
        if step is None:
            step = ''
        else:
            step = '_step'+str(step)
        if keys is None:
            keys = self.network_keys_to_save
        assert keys is not None
        for key in keys:
            T.save(self.network_dict[key].state_dict(), self.ckpt_path+'/ckpt_'+key+ep+step+'.pt')

    def _load_network(self, keys=None, ep=None, step=None):
        if (not self.non_flat_obs) and self.observation_normalization:
            self.normalizer.history_mean = np.load(os.path.join(self.data_path, 'input_means.npy'))
            self.normalizer.history_var = np.load(os.path.join(self.data_path, 'input_vars.npy'))
        if ep is None:
            ep = ''
        else:
            ep = '_ep'+str(ep)
        if step is None:
            step = ''
        else:
            step = '_step'+str(step)
        if keys is None:
            keys = self.network_keys_to_save
        assert keys is not None
        for key in keys:
            self.network_dict[key].load_state_dict(T.load(self.ckpt_path+'/ckpt_'+key+ep+step+'.pt', map_location=self.device))

    def _save_statistics(self, keys=None):
        if (not self.non_flat_obs) and self.observation_normalization:
            np.save(os.path.join(self.data_path, 'input_means'), self.normalizer.history_mean)
            np.save(os.path.join(self.data_path, 'input_vars'), self.normalizer.history_var)
        if keys is None:
            keys = self.statistic_dict.keys()
        for key in keys:
            if len(self.statistic_dict[key]) == 0:
                continue
            # convert everything to a list before save via json
            if T.is_tensor(self.statistic_dict[key][0]):
                self.statistic_dict[key] = T.as_tensor(self.statistic_dict[key], device=self.device).cpu().numpy().tolist()
            else:
                self.statistic_dict[key] = np.array(self.statistic_dict[key]).tolist()
            json.dump(self.statistic_dict[key], open(os.path.join(self.data_path, key+'.json'), 'w'))
    
    def _plot_statistics(self, keys=None, x_labels=None, y_labels=None, window=5, save_to_file=True):
        if save_to_file:
            self._save_statistics(keys=keys)
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
                if ('loss' in key) or ('alpha' in key) or ('entropy' in key) or ('step' in key):
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
                if len(self.statistic_dict[key]) == 0:
                    continue
                smoothed_plot(os.path.join(self.path, key+'.png'), self.statistic_dict[key],
                              x_label=x_labels[key], y_label=y_labels[key], window=window)
        else:
            for key in keys:
                smoothed_plot(os.path.join(self.path, key+'.png'), self.statistic_dict[key],
                              x_label=x_labels[key], y_label=y_labels[key], window=window)

