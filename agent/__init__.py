from agent.ddpg import DDPG
from agent.ddpg_goal_conditioned import GoalConditionedDDPG
from agent.sac import SAC
from agent.sac_goal_conditioned import GoalConditionedSAC

from agent.dqn_atari import DQN
from agent.dqn_her_discrete import HindsightDQN
from agent.option_critic_discrete import OptionCritic
from agent.option_critic_her_continuous import HindsightOptionCritic
from agent.option_dqn_her_discrete import OptionDQN
from agent.ppo_continuous import PPOAgent

agents = {
    'ddpg': DDPG,
    'ddpg_her': GoalConditionedDDPG,
    'sac': SAC,
    'sac_her': GoalConditionedSAC,

    'ppo': PPOAgent,
    'dqn_atari': DQN,
    'dqn_her': HindsightDQN,
    'option_critic': OptionCritic,
    'option_critic_her': HindsightOptionCritic,
    'option_critic_dqn': OptionDQN,
}
