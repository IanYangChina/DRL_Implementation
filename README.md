## DRL_Implementation
#### Current status: under development  
#### Important: the project is undergoing an code structure overhaul, use with caution

#### Introduction
- This repository is a pytorch-based implementation of modern DRL algorithms, designed to be reusable for as many 
Gym-like training environments as possible
- The package is under-development, plan to release the first version before April 2021.
- Language: Python >= 3.6
- Main library: PyTorch-1.3.0
- Tested environments: Gym, Pybullet-gym, Pybullet-multigoal-gym

#### Algorithms: Flat
- [ ] DQN - Deterministic, Discrete (LSTM network for Atari)
- [X] DDPG - Deterministic, Continuous
- [ ] PPO - Stochastic, Continuous
- [X] TD3 -Deterministic, Discrete
- [X] SAC (Adaptive Temperature) - Stochastic, Continuous

#### Algorithms: Distributed
- [ ] D4PG - Deterministic, Continuous
- [ ] R2D2 - Deterministic, Discrete

#### Algorithms: Hierarchical
- [X] Option DQN - Hindsight, Deterministic, Discrete
- [X] Option Critic - Hindsight, Stochastic, Discrete & Continuous
- [ ] HIRO - Hindsight, Deterministic, Continuous
- [ ] HAC - Hindsight, Deterministic, Continuous

#### Replay buffers
- [X] Hindsight
- [X] Prioritised

#### Environments
- [X] GridWorld_MultiRoomKeyDoor (Discrete, Multi-goal, Customized)
- [X] OpenAI Gym Mujoco Robotics Multigoal Environment (Continuous, Official)
- [X] [Pybullet Multigoal Gym](https://github.com/IanYangChina/pybullet_multigoal_gym) (OpenAI Robotics 
Multigoal Pybullet Migration) (Continuous, Official)
- [ ] OpenAI Gym Mujoco Robotic Multi-goal/task/stage Environment (Continuous, Customized)

#### Some result figures
<img src="/src/returns_pybullet_kuka_reach.png" width="350"/>
<img src="/src/returns_half_cheetah.png" width="350"/>
<img src="/src/returns_pendulum.png" width="350"/>

#### Reference Papers: Algorithm
* [DQN](https://www.nature.com/articles/nature14236?wm=book_wap_0005)
* [DoubleDQN](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389)
* [LSTM network on raw Atari pixel observation](https://arxiv.org/pdf/1907.02908.pdf)
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

#### Reference Papers: Implementation Matters
* [Time limit](https://arxiv.org/abs/1712.00378)
* [SOTA PPO Hyperparameters (many applicable to other algorithms)](https://arxiv.org/abs/2006.05990)
* [SAC Temperature Auto-tuning](https://arxiv.org/abs/1812.05905)

#### Reference Repos
* [RL-Adventure-DDPG by higgsfield](https://github.com/higgsfield/RL-Adventure-2/blob/master/5.ddpg.ipynb)
* [OpenAI HER Baseline](https://github.com/openai/baselines/tree/master/baselines/her)
* [hindsight-experience-replay by TianhongDai](https://github.com/TianhongDai/hindsight-experience-replay)
* [rlkit](https://github.com/vitchyr/rlkit)
* [pybullet multigoal gym](https://github.com/IanYangChina/pybullet_multigoal_gym)
