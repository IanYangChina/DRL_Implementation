from .agent.continuous_action.ddpg import DDPG
from .agent.continuous_action.ddpg_goal_conditioned import GoalConditionedDDPG
from .agent.continuous_action.sac import SAC
from .agent.continuous_action.sac_goal_conditioned import GoalConditionedSAC
from .agent.continuous_action.td3 import TD3
from .agent.continuous_action.distributional_ddpg import DistributionalDDPG
from .agent.continuous_action.d4pg import D4PG
from .agent.continuous_action.sac_DrQ import SACDrQ

agents = {
    'ddpg': DDPG,
    'ddpg_her': GoalConditionedDDPG,
    'sac': SAC,
    'sac_her': GoalConditionedSAC,
    'td3': TD3,
    'distri_ddpg': DistributionalDDPG,
    'd4pg': D4PG
}
