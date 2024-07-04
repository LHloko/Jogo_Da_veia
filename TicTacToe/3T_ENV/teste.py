import glob
import os
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
from pettingzoo.classic import tictactoe_v3

def jogar_ai_namoral(env_fn, num_games=3, render_mode='human', **env_kwargs):
    
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    
    zap0 = ['tictactoe_v3_20240629-160301', 'tictactoe_v3_20240628-073726', 'tictactoe_v3_20240628-041046']
    zap1 = ['tictactoe_v3_20240625-205125', 'tictactoe_v3_20240626-013610', 'tictactoe_v3_20240626-041152']
    zap3 = ['tictactoe_v3_20240626-175843','tictactoe_v3_20240626-212937','tictactoe_v3_20240627-092421']
    
    model = MaskablePPO.load(zap1[1])

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            observation, action_mask = obs.values()

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[
                        winner
                    ]  # only tracks the largest reward (winner of game)
                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    act = int(input('de a jogada fellas '))
                    # act = env.action_space(agent).sample(action_mask)
                else:
                    # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )

            env.step(act)
            env.render()
            
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores

if __name__ == "__main__":
    env_fn = tictactoe_v3

    env_kwargs = {}

    # Eu jogando contra meu fellas agente
    jogar_ai_namoral(env_fn, num_games=3, render_mode="luiz", **env_kwargs)