from agent.continuous_action.ddpg import DDPG
from agent.continuous_action.ddpg_goal_conditioned import GoalConditionedDDPG
from agent.continuous_action.sac import SAC
from agent.continuous_action.sac_goal_conditioned import GoalConditionedSAC
from agent.continuous_action.ppo import PPO
from agent.continuous_action.td3 import TD3
# from agent.continuous_action.option_critic_her_continuous import HindsightOptionCritic

from agent.discrete_action.dqn_atari import DQN
# from agent.discrete_action.dqn_her_discrete import HindsightDQN
# from agent.discrete_action.option_critic_discrete import OptionCritic
# from agent.discrete_action.option_dqn_her_discrete import OptionDQN

agents = {
    'ddpg': DDPG,
    'ddpg_her': GoalConditionedDDPG,
    'sac': SAC,
    'sac_her': GoalConditionedSAC,
    'ppo': PPO,

    'dqn_atari': DQN,
    # 'dqn_her': HindsightDQN,
    # 'option_critic': OptionCritic,
    # 'option_critic_her': HindsightOptionCritic,
    # 'option_critic_dqn': OptionDQN,
}
