import os
import time
import torch as T
import numpy as np
import json
import queue
import importlib
import multiprocessing as mp
from collections import namedtuple
from .utils.plot import smoothed_plot
from .utils.replay_buffer import ReplayBuffer, PrioritisedReplayBuffer
from .utils.normalizer import Normalizer
# T.multiprocessing.set_start_method('spawn')
t = namedtuple("transition", ('state', 'action', 'next_state', 'reward', 'done'))


def mkdir(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


class Agent(object):
    def __init__(self, algo_params, image_obs=False, action_type='continuous', path=None, seed=-1):
        # torch device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        if 'cuda_device_id' in algo_params.keys():
            self.device = T.device("cuda:%i" % algo_params['cuda_device_id'])
        # path & seeding
        T.manual_seed(seed)
        T.cuda.manual_seed_all(seed)  # this has no effect if cuda is not available

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
        if self.image_obs:
            self.state_dim = 0
            self.state_shape = algo_params['state_shape']
        else:
            self.state_dim = algo_params['state_dim']
        self.action_dim = algo_params['action_dim']
        if self.action_type == 'continuous':
            self.action_max = algo_params['action_max']
            self.action_scaling = algo_params['action_scaling']

        # common args
        if not self.image_obs:
            # todo: observation in distributed training should be synced as well
            self.observation_normalization = algo_params['observation_normalization']
            # if not using obs normalization, the normalizer is just a scale multiplier, returns inputs*scale
            self.normalizer = Normalizer(self.state_dim,
                                         algo_params['init_input_means'], algo_params['init_input_vars'],
                                         activated=self.observation_normalization)

        self.gamma = algo_params['discount_factor']
        self.tau = algo_params['tau']

        # network dict is filled in each specific agent
        self.network_dict = {}
        self.network_keys_to_save = None

        # algorithm-specific statistics are defined in each agent sub-class
        self.statistic_dict = {
            # use lowercase characters
            'actor_loss': [],
            'critic_loss': [],
        }

    def _soft_update(self, source, target, tau=None, from_params=False):
        if tau is None:
            tau = self.tau

        if not from_params:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )
        else:
            for target_param, param in zip(target.parameters(), source):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + T.tensor(param).float().to(self.device) * tau
                )

    def _save_network(self, keys=None, ep=None):
        if ep is None:
            ep = ''
        else:
            ep = '_ep' + str(ep)
        if keys is None:
            keys = self.network_keys_to_save
        assert keys is not None
        for key in keys:
            T.save(self.network_dict[key].state_dict(), self.ckpt_path + '/ckpt_' + key + ep + '.pt')

    def _load_network(self, keys=None, ep=None):
        if not self.image_obs:
            self.normalizer.history_mean = np.load(os.path.join(self.data_path, 'input_means.npy'))
            self.normalizer.history_var = np.load(os.path.join(self.data_path, 'input_vars.npy'))
        if ep is None:
            ep = ''
        else:
            ep = '_ep' + str(ep)
        if keys is None:
            keys = self.network_keys_to_save
        assert keys is not None
        for key in keys:
            self.network_dict[key].load_state_dict(T.load(self.ckpt_path + '/ckpt_' + key + ep + '.pt'))

    def _save_statistics(self, keys=None):
        if not self.image_obs:
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
                smoothed_plot(os.path.join(self.path, key + '.png'), self.statistic_dict[key],
                              x_label=x_labels[key], y_label=y_labels[key], window=window)
        else:
            for key in keys:
                smoothed_plot(os.path.join(self.path, key + '.png'), self.statistic_dict[key],
                              x_label=x_labels[key], y_label=y_labels[key], window=window)


class Worker(Agent):
    def __init__(self, algo_params, queues, path=None, seed=0, i=0):
        self.queues = queues
        self.worker_id = i
        self.worker_update_gap = algo_params['worker_update_gap']  # in episodes
        self.env_step_count = 0
        super(Worker, self).__init__(algo_params, path=path, seed=seed)

    def run(self, render=False, test=False, load_network_ep=None, sleep=0):
        raise NotImplementedError

    def _interact(self, render=False, test=False, sleep=0):
        raise NotImplementedError

    def _select_action(self, obs, test=False):
        raise NotImplementedError

    def _remember(self, batch):
        try:
            self.queues['replay_queue'].put_nowait(batch)
        except queue.Full:
            pass

    def _download_actor_networks(self, keys, tau=1.0):
        try:
            source = self.queues['network_queue'].get_nowait()
        except queue.Empty:
            return False
        print("Worker No. %i downloading network" % self.worker_id)
        for key in keys:
            self._soft_update(source[key], self.network_dict[key], tau=tau, from_params=True)
        return True


class Learner(Agent):
    def __init__(self, algo_params, queues, path=None, seed=0):
        self.queues = queues
        self.num_workers = algo_params['num_workers']
        self.learner_steps = algo_params['learner_steps']
        self.learner_upload_gap = algo_params['learner_upload_gap']  # in optimization steps
        self.actor_learning_rate = algo_params['actor_learning_rate']
        self.critic_learning_rate = algo_params['critic_learning_rate']
        self.discard_time_limit = algo_params['discard_time_limit']
        self.batch_size = algo_params['batch_size']
        self.prioritised = algo_params['prioritised']
        self.optimizer_steps = algo_params['optimization_steps']
        self.optim_step_count = 0
        super(Learner, self).__init__(algo_params, path=path, seed=seed)

    def run(self):
        raise NotImplementedError

    def _learn(self, steps=None):
        raise NotImplementedError

    def _upload_learner_networks(self, keys):
        print("Learner uploading network")
        params = dict.fromkeys(keys)
        for key in keys:
            params[key] = [p.data.cpu().detach().numpy() for p in self.network_dict[key].parameters()]
        # delete an old net and upload a new one
        try:
            data = self.queues['network_queue'].get_nowait()
            del data
        except queue.Empty:
            pass
        try:
            self.queues['network_queue'].put(params)
        except queue.Full:
            pass


class CentralProcessor(object):
    def __init__(self, algo_params, env_name, env_source, learner, worker, transition_tuple=None, path=None,
                 worker_seeds=None, seed=0):
        self.algo_params = algo_params.copy()
        self.env_name = env_name
        assert env_source in ['gym', 'pybullet_envs', 'pybullet_multigoal_gym'], \
            "unsupported env source: {}, " \
            "only 3 env sources are supported: {}, " \
            "for new env sources please modify the original code".format(env_source,
                                                                         ['gym', 'pybullet_envs',
                                                                          'pybullet_multigoal_gym'])
        self.env_source = importlib.import_module(env_source)
        self.learner = learner
        self.worker = worker
        self.batch_size = algo_params['batch_size']
        self.num_workers = algo_params['num_workers']
        self.learner_steps = algo_params['learner_steps']
        if worker_seeds is None:
            worker_seeds = np.random.randint(10, 1000, size=self.num_workers).tolist()
        else:
            assert len(worker_seeds) == self.num_workers, 'should assign seeds to every worker'
        self.worker_seeds = worker_seeds
        assert path is not None, 'please specify a project path to save files'
        self.path = path
        # create a random number generator and seed it
        self.rng = np.random.default_rng(seed=0)

        # multiprocessing queues
        self.queues = {
            'replay_queue': mp.Queue(maxsize=algo_params['replay_queue_size']),
            'batch_queue': mp.Queue(maxsize=algo_params['batch_queue_size']),
            'network_queue': T.multiprocessing.Queue(maxsize=self.num_workers),
            'learner_step_count': mp.Value('i', 0),
            'global_episode_count': mp.Value('i', 0),
        }

        # setup replay buffer
        # prioritised replay
        self.prioritised = algo_params['prioritised']
        self.store_with_given_priority = algo_params['store_with_given_priority']
        # non-goal-conditioned replay buffer
        tr = transition_tuple
        if transition_tuple is None:
            tr = t
        if not self.prioritised:
            self.buffer = ReplayBuffer(algo_params['memory_capacity'], tr, seed=seed)
        else:
            self.queues.update({
                'priority_queue': mp.Queue(maxsize=algo_params['priority_queue_size'])
            })
            self.buffer = PrioritisedReplayBuffer(algo_params['memory_capacity'], tr, rng=self.rng)

    def run(self):
        def worker_process(i, seed):
            env = self.env_source.make(self.env_name)
            path = os.path.join(self.path, "worker_%i" % i)
            worker = self.worker(self.algo_params, env, self.queues, path=path, seed=seed, i=i)
            worker.run()
            self.empty_queue('replay_queue')

        def learner_process():
            env = self.env_source.make(self.env_name)
            path = os.path.join(self.path, "learner")
            learner = self.learner(self.algo_params, env, self.queues, path=path, seed=0)
            learner.run()
            if self.prioritised:
                self.empty_queue('priority_queue')
            self.empty_queue('network_queue')

        def update_buffer():
            while self.queues['learner_step_count'].value < self.learner_steps:
                num_transitions_in_queue = self.queues['replay_queue'].qsize()
                for n in range(num_transitions_in_queue):
                    data = self.queues['replay_queue'].get()
                    if self.prioritised:
                        if self.store_with_given_priority:
                            self.buffer.store_experience_with_given_priority(data['priority'], *data['transition'])
                        else:
                            self.buffer.store_experience(*data)
                    else:
                        self.buffer.store_experience(*data)
                if self.batch_size > len(self.buffer):
                    continue

                if self.prioritised:
                    try:
                        inds, priorities = self.queues['priority_queue'].get_nowait()
                        self.buffer.update_priority(inds, priorities)
                    except queue.Empty:
                        pass
                    try:
                        batch, weights, inds = self.buffer.sample(batch_size=self.batch_size)
                        state, action, next_state, reward, done = batch
                        self.queues['batch_queue'].put_nowait((state, action, next_state, reward, done, weights, inds))
                    except queue.Full:
                        continue
                else:
                    try:
                        batch = self.buffer.sample(batch_size=self.batch_size)
                        state, action, next_state, reward, done = batch
                        self.queues['batch_queue'].put_nowait((state, action, next_state, reward, done))
                    except queue.Full:
                        time.sleep(0.1)
                        continue

            self.empty_queue('batch_queue')

        processes = []
        p = T.multiprocessing.Process(target=update_buffer)
        processes.append(p)
        p = T.multiprocessing.Process(target=learner_process)
        processes.append(p)
        for i in range(self.num_workers):
            p = T.multiprocessing.Process(target=worker_process,
                                          args=(i, self.worker_seeds[i]))
            processes.append(p)

        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def empty_queue(self, queue_name):
        while True:
            try:
                data = self.queues[queue_name].get_nowait()
                del data
            except queue.Empty:
                break
        self.queues[queue_name].close()
