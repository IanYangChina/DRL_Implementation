## DRL_Implementation
##### Current status: minimal updates

### Introduction
- This repository is a pytorch-based implementation of modern DRL algorithms, designed to be reusable for as many 
Gym-like training environments as possible
- The package is mainly for my personal usage, however feel free to use it as you like.
- It is recommended to use the [released version](https://github.com/IanYangChina/DRL_Implementation/tree/v2.0)
- Understand more with the [Wiki!](https://github.com/IanYangChina/DRL_Implementation/wiki)
- Tested environments: Gym, Pybullet-gym, Pybullet-multigoal-gym
- *My priority is on continuous action algorithms as I'm working on robotics*

#### Installation
```
git clone https://github.com/IanYangChina/DRL_Implementation.git
cd DRL_Implementation
python -m pip install -r requirements.txt
python -m pip install .
```
[Click here for example codes](https://github.com/IanYangChina/DRL_Implementation/tree/master/drl_implementation/examples)
, to run the codes you will need to install Gym, Pybullet, or pybullet-multigoal-gym. See env installation links below.
For more use cases, have a look at the [drl_imp_test repo](https://github.com/IanYangChina/drl_imp_test)\
From the project root, run `python drl_implementation/examples/$SCTIPT_NAME.py`

##### State-based
- [X] Distributional DDPG, Continuous
- [X] DDPG - Deterministic, Continuous
- [X] TD3 -Deterministic, Continuous
- [X] SAC (Adaptive Temperature) - Stochastic, Continuous

##### Replay buffers
- [X] Hindsight
- [X] Prioritised

##### Tested Environments
- [X] [Pybullet Gym (Continuous)](https://github.com/bulletphysics/bullet3)
- [X] [OpenAI Gym Mujoco Robotics Multigoal Environment (Continuous)](https://openai.com/blog/ingredients-for-robotics-research/)
- [X] [Pybullet Multigoal Gym](https://github.com/IanYangChina/pybullet_multigoal_gym) (OpenAI Robotics 
Multigoal Pybullet Migration) (Continuous)

##### Some result figures
<img src="/src/figs.png" width="600"/>
<img src="/src/push.gif" width="600"/>
<img src="/src/pendulum.gif" width="600"/>

#### Reference Papers: Algorithm
* [DQN](https://www.nature.com/articles/nature14236?wm=book_wap_0005)
* [DoubleDQN](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389)
* [DDPG](https://arxiv.org/abs/1509.02971)
* [TD3](https://arxiv.org/pdf/1802.09477.pdf)
* [SAC (Adaptive Temperature)](https://arxiv.org/pdf/1812.05905.pdf)
* [PER](https://arxiv.org/abs/1511.05952)
* [HER](http://papers.nips.cc/paper/7090-hindsight-experience-replay)

#### Reference Papers: Implementation Matters
* [Time limit](https://arxiv.org/abs/1712.00378)
* [SOTA PPO Hyperparameters (many applicable to other algorithms)](https://arxiv.org/abs/2006.05990)
* [SAC Temperature Auto-tuning](https://arxiv.org/abs/1812.05905)
