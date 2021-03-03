import pybullet_envs
import matplotlib.pyplot as plt
from drl_implementation.agent.utils.env_wrapper import PixelPybulletGym, FrameStack
import torch as T
device = T.device("cuda" if T.cuda.is_available() else "cpu")

li = [T.tensor([1.0], device=device), T.tensor([1.0], device=device), T.tensor([1.0], device=device)]

li_t = T.tensor(li, device=device)
print(li_t)