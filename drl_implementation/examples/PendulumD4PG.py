import os
import drl_implementation

algo_params = {
    'prioritised': True,
    'store_with_given_priority': False,
    'memory_capacity': int(1e6),
    'actor_learning_rate': 0.001,
    'critic_learning_rate': 0.001,
    'Q_weight_decay': 0.0,
    'update_interval': 1,
    'batch_size': 100,
    'optimization_steps': 1,
    'tau': 0.005,
    'discount_factor': 0.99,
    'discard_time_limit': False,
    'warmup_step': 2500,
    'observation_normalization': False,
    'num_atoms': 51,
    'value_max': 50,
    'value_min': -50,
    'reward_scaling': 1,

    'num_workers': 4,
    'learner_steps': int(1e6),
    'learner_upload_gap': int(1e3),
    'worker_update_gap': 3,
    'replay_queue_size': 64,
    'priority_queue_size': 64,
    'batch_queue_size': 10,

    'training_episodes': 101,
    'testing_gap': 10,
    'testing_episodes': 10,
    'saving_gap': 50,
}

if __name__ == '__main__':
    agent = drl_implementation.D4PG(algo_params,
                                    env_name="InvertedPendulumSwingupBulletEnv-v0",
                                    env_source="pybullet_envs",
                                    path=os.path.dirname(os.path.realpath(__file__))+'/d4pg')
    agent.run()
