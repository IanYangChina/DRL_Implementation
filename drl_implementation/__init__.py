from .agent.continuous_action.ddpg import DDPG
from .agent.continuous_action.ddpg_goal_conditioned import GoalConditionedDDPG
from .agent.continuous_action.sac import SAC
from .agent.continuous_action.sac_goal_conditioned import GoalConditionedSAC
from .agent.continuous_action.td3 import TD3
from .agent.continuous_action.distributional_ddpg import DistributionalDDPG

agents = {
    'ddpg': DDPG,
    'ddpg_her': GoalConditionedDDPG,
    'sac': SAC,
    'sac_her': GoalConditionedSAC,
    'td3': TD3,
    'distri_ddpg': DistributionalDDPG,
}
