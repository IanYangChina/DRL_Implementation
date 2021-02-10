import os
import drl_implementation

algo_params = {
    'prioritised': True,
    'store_with_given_priority': False,
    'memory_capacity': int(1e6),
    'actor_learning_rate': 0.001,
    'critic_learning_rate': 0.001,
    'Q_weight_decay': 0.0,
    'batch_size': 128,
    'optimization_steps': 1,
    'tau': 0.001,
    'discount_factor': 0.99,
    'discard_time_limit': False,
    'observation_normalization': False,
    'num_atoms': 51,
    'value_max': 1000,
    'value_min': -1000,
    'reward_scaling': 1,
    'gaussian_scale': 0.3,
    'gaussian_sigma': 1.0,

    'num_workers': 4,
    'learner_steps': int(1e6),
    'learner_upload_gap': int(1e3),
    'worker_update_gap': 3,
    'replay_queue_size': 64,
    'priority_queue_size': 64,
    'batch_queue_size': 64,

    'training_episodes': 101,
    'testing_gap': 10,
    'testing_episodes': 10,
    'saving_gap': 50,
}

agent = drl_implementation.D4PG(algo_params,
                                env_name="InvertedPendulumSwingupBulletEnv-v0",
                                env_source="pybullet_envs",
                                path=os.path.dirname(os.path.realpath(__file__)))
agent.run()
