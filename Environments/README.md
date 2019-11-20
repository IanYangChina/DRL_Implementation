# Environments  
  
### KeyDoorEnvironment  
- **Brief Introduction**:  
    An grid-world constituted by three types of rooms: a hall, middle rooms and final rooms.  
    When calling function reset(), an agent will be initialized at a random cell within the hall.  
    In order to enter middle rooms or final rooms, an agent needs to first collect the corresponding keys.  
    Keys for entering middle rooms are randomly placed within the hall.  
    Keys for entering final rooms are randomly placed within middle rooms.  
    The ultimate goal-locations are cells of the final rooms.  
    
- **MDP Definition**:  
    States and goals are represented by [x, y] coordinates of the cells.  
    Observation is constituted by a state, a goal and an inventory of keys.  
        The inventory is a one-hot vector whose size is the number of keys in an environment instance.  
    Primitive actions are "up", "down", "left" and "right" (deterministic actions). Reward for primitive-agent is 1 when a desired goal given by the option-agent is achieved and 0 otherwise.  
    Options are coordinates of goals. Reward for option-agent is 1 when an ultimate goal is achieved and 0 otherwise.  
- **Challenge**:  
  **Exploration efficiency**  
    This is due to the inter-connections between goals (keys and doors).  
    Specifically, consider the simplest instance with 1 middle and final room, the agent needs to reach 4 goals (middle key, middle door, final key, final door) in a specific order, before it has a chance to reach the ultimate goal.  
    Since the difficulty of entering a room is increasing from middle rooms to final rooms, the exploration of rooms are also increasing.  
    Thus, achieving an ultimate goal is fairly difficult, and it is necessary to introduce better ideas that improve exploration instead of solely counting on random actions.  
  **Sample efficiency**  
    This is a common issue bothering RL algorithms in sparse reward environments.  
    However, in this environment where goals are inter-related to each other, obtaining experiences valuable for learning is even more unusual.  
    Note that, Hindsight Experience Replay is helpful for training the low level agent, while it is not so for training the high level agent. Since the reward is only obtained when achieving an ultimate goal, not an arbitrary goal as the case of low level agent.
