#### Some tips

To run the FetchReach-v1 environment, you will need to install gym, mujoco and mujoco-py on your environment.
Here's the links:  

[OpenAI Gym](https://github.com/openai/gym)  
[Get a free Mujoco trial license, here](https://www.roboti.us/license.html)  
[Install Mujoco and Mujoco.py by OpenAI, here](https://github.com/openai/mujoco-py)  

Unfortunately, it seems that mujoco is hard to be installed on Windows systems, but you might find some help in [this
page](https://github.com/openai/mujoco-py/issues/253). However, I still suggest you to run these codes on Linux systems.

The [pybullet-multigoal-gym](https://github.com/IanYangChina/pybullet_multigoal_gym) environment is a migration of the open ai gym multigoal environment, developed by the author
of this repo. It is free as it is based on Pybullet. You will need it to run the pybullet experiments.

#### Some notes I made when I was implementing HER:  
* The original paper uses multiple cpu to collect data, however, this implementation uses single cpu. Multi-cpu might be 
added in the future.  
* Actor, critic networks have 3 hidden layers, each with 256 units and relu activation; critic output without activation, 
while actor output with tanh and rescaling.  
* Observation and goal are concatenated and fed into both networks.  
* The original paper scales observation, goals and actions into [-5, 5] (we don't need rescaling with the Gym environment), 
and normalize to 0 mean and standard variation. The means and standard deviations are computed using encountered data.  
* Training process has 200 epochs with 50 cycles, each of which has 16 episodes and 40 optimization steps. The total 
episode number is 200\*50\*16=160000, each of which has 50 time steps. After every 16 episodes, 40 optimization steps are 
performed.  
* Each optimization step uses a mini-batch of 128 batch size uniformly sampled from a replay buffer with 10^6 capacity,
target network is updated softly with tau=0.05.  
* Adam is used for learning with a learning rate of 0.001, discount factor is 0.98, target value is clipped to 
[-1/(1-0.98), 0], that is [-50, 0]. I think this is based on the 50 time steps they set for each episode, in which at 
most an agent could gain -50 return.
* For exploration, they randomly select action from uniform distribution with 20% chance; and with 80% chance, they 
add normal noise into increments along each axes with standard deviation equal to 5% of the max bound.

* The SAC agent doesn't need a behavioural policy.
* The goal-conditioned **sac** agent doesn't need value clipping.
* Prioritised replay supported.
    
#### Results on the task 'Push'
<img src="./push.gif" width="400"/>
