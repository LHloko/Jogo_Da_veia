from ttt_env import T3Env
import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

import glob
import os
import time

def mask_fn(env: gym.Env):
    valid_action = np.zeros(9)
    for i in env.get_mask_action():
        valid_action[i]=1
    return valid_action

env = T3Env()


env = ActionMasker(env, mask_fn)


model = MaskablePPO(
    policy = MaskableActorCriticPolicy,
    env = env,
    learning_rate=0.0003,
    gamma=0.99,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    rollout_buffer_class=None,
    rollout_buffer_kwargs=None,
    target_kl=None,
    stats_window_size=100,
    tensorboard_log="./mppo_tensorboard/",
    verbose=0,
    )

#model = MaskablePPO.load("MPPO_rand_20240703-01_00_02", env)

model.learn(total_timesteps=50000000,
            progress_bar=True, 
            tb_log_name="e_0_005_gm_0_99_lr_0_0003",
            reset_num_timesteps=False)

model.save("MPPO_teste_01")

print("Model has been saved.")

'''
model = MaskablePPO.load("MPPO_20240702-14_28_54")
for i in range(5):
    obs, _ = env.reset()
    done = False
    
    while done != True:
        action_masks = get_action_masks(env)
        print(action_masks)
        action, _states = model.predict(obs, action_masks=action_masks)
        print(action)
        obs, reward, terminated, truncated, info = env.play_human(action)
        print('obs',obs)
        print('info',info)
        print('reward',reward)
        print(terminated)
        print(truncated)
        done = terminated

    print('======================')
'''
    