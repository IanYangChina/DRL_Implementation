from agent.ddpg import DDPG
from agent.ddpg_her_continuous import HindsightDDPGAgent
from agent.dqn_atari import DQN
from agent.dqn_her_discrete import HindsightDQN
from agent.option_critic_discrete import OptionCritic
from agent.option_critic_her_continuous import HindsightOptionCritic
from agent.option_dqn_her_discrete import OptionDQN
from agent.ppo_continuous import PPOAgent
from agent.sac_continuous import SACAgent
from agent.sac_her_continuous import HindsightSACAgent

agents = {
    'ddpg': DDPG,
    'ddpg_her': HindsightDDPGAgent,
    'dqn_atari': DQN,
    'dqn_her': HindsightDQN,
    'option_critic': OptionCritic,
    'option_critic_her': HindsightOptionCritic,
    'option_critic_dqn': OptionDQN,
    'ppo': PPOAgent,
    'sac': SACAgent,
    'sac_her': HindsightSACAgent
}
