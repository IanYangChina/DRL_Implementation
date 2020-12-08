## DRL_Implementation
- This repository will be filled with codes reproducing some DRL algos I'm interested in.
- Most algorithms remain untested because my PC is occupied by my current research experiments.
- Language: Python-3.6
- Main library: PyTorch-1.3.0, Mujoco-py-2.0.2.8, Gym-0.15.3  

#### Current status: under development  

#### Algorithms: Flat
- [X] DQN - Deterministic, Discrete
- [X] DDPG - Deterministic, Continuous
- [X] TD3 -Deterministic, Discrete
- [X] SAC (Adaptive Temperature) - Stochastic, Continuous

#### Algorithms: Distributed
- [ ] D4PG - Deterministic, Continuous
- [ ] R2D2 - Deterministic, Discrete

#### Algorithms: Hierarchical
- [X] Option DQN - Hindsight, Deterministic, Discrete
- [ ] Option DDPG - Hindsight, Deterministic, Continuous
- [X] Option Critic - Hindsight, Stochastic, Discrete & Continuous
- [ ] HIRO - Hindsight, Deterministic, Continuous
- [ ] HAC - Hindsight, Deterministic, Continuous

#### Replay buffers:
- [X] Hindsight
- [X] Prioritised

#### Environments
- [X] GridWorld_MultiRoomKeyDoor (Discrete, Multi-goal, Customized)
- [X] OpenAI Gym Mujoco Robotics Multigoal Environment (Continuous, Official)
- [X] [Pybullet Multigoal Gym](https://github.com/IanYangChina/pybullet_multigoal_gym) (OpenAI Robotics Multigoal Pybullet Migration) (Continuous, Official)
- [ ] OpenAI Gym Mujoco Robotic Multi-goal/task/stage Environment (Continuous, Customized)

#### Reference Papers
* [DQN](https://www.nature.com/articles/nature14236?wm=book_wap_0005)
* [DoubleDQN](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [TD3](https://arxiv.org/pdf/1802.09477.pdf)
* [SAC (Adaptive Temperature)](https://arxiv.org/pdf/1812.05905.pdf)
* [PER](https://arxiv.org/abs/1511.05952)
* [HER](http://papers.nips.cc/paper/7090-hindsight-experience-replay)
* [HIRO](http://papers.nips.cc/paper/7591-data-efficient-hierarchical-reinforcement-learning.pdf)
* [HAC](https://arxiv.org/abs/1712.00948)
* [OptionFramework](https://www.sciencedirect.com/science/article/pii/S0004370299000521)
* [OptionCritic](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14858)
* [D4PG](https://arxiv.org/abs/1804.08617)
* [R2D2](https://openreview.net/pdf?id=r1lyTjAqYX)

#### Reference Repos
* [RL-Adventure-DDPG by higgsfield](https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb)
* [OpenAI HER Baseline](https://github.com/openai/baselines/tree/master/baselines/her)
* [hindsight-experience-replay by TianhongDai](https://github.com/TianhongDai/hindsight-experience-replay)
* [rlkit](https://github.com/vitchyr/rlkit)
* [pybullet multigoal gym](https://github.com/IanYangChina/pybullet_multigoal_gym)
