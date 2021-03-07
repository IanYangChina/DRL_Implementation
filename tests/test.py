import pybullet_envs
import matplotlib.pyplot as plt
from drl_implementation.agent.utils.env_wrapper import PixelPybulletGym, FrameStack
import torch as T
import numpy as np
import time
device = T.device("cuda")
a = ((2.0, 2.0, 2.20, 2.0, 2.2, 3.333),
     (2.0, 2.0, 2.20, 2.0, 2.2, 3.333),
     (2.0, 2.0, 2.20, 2.0, 2.2, 3.333),
     (2.0, 2.0, 2.20, 2.0, 2.2, 3.333),
     (2.0, 2.0, 2.20, 2.0, 2.2, 3.333),
     (2.0, 2.0, 2.20, 2.0, 2.2, 3.333),
     (2.0, 2.0, 2.20, 2.0, 2.2, 3.333),
     (2.0, 2.0, 2.20, 2.0, 2.2, 3.333),
     (2.0, 2.0, 2.20, 2.0, 2.2, 3.333))

start = time.time()
b = T.tensor([a], device=device)
print(time.time() - start)
print(b.size())

start = time.time()
c = T.as_tensor([np.array(a)], device=device)
print(time.time() - start)
print(c.size())

start = time.time()
d = T.as_tensor([a], device=device)
print(time.time() - start)
print(d.size())