from ttt_env import T3Env
import gymnasium as gym
import numpy as np
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

def mask_fn(env: gym.Env):
    valid_action = np.zeros(9)
    for i in env.get_mask_action():
        valid_action[i]=1
    return valid_action

env = T3Env('human')


env = ActionMasker(env, mask_fn)

models = ['MPPO_20240702-14_28_54', # Ótimo
          'MPPO_20240702-15_10_02', # Ruim
          'MPPO_rand_20240702-17_08_02', # Ótimo
          'MPPO_rand_20240702-17_36_26', # Ótimo
          'MPPO_rand_20240702-17_08_02', # ???
          'MPPO_rand_20240703-01_00_02',
          'MPPO_rand3_20240703-16_44_58',
          ]

model = MaskablePPO.load('MPPO_teste_01')
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
for _ in range(5):
    act = int(input('play ai fellas '))
    env.play_human_first(env, act)
'''

