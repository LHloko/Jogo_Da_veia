import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import random

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks


class Tabuleiro:
    def __init__(self):

        # empty -- 0
        # player 0 -- 1
        # player 1 -- 2
        self.squares = [0] * 9

        # precommute possible winning combinations
        self.calculate_winners()

    def setup(self):
        self.calculate_winners()

    def play_turn(self, agent, pos):
        # if spot is empty
        if self.squares[pos] != 0:
            return
        
        if agent == 0: # X 
            self.squares[pos] = 1 
        elif agent == 1: # O
            self.squares[pos] = 2
            
        return

    def calculate_winners(self):
        winning_combinations = []
        indices = [x for x in range(0, 9)]

        # Vertical combinations
        winning_combinations += [
            tuple(indices[i : (i + 3)]) for i in range(0, len(indices), 3)
        ]

        # Horizontal combinations
        winning_combinations += [
            tuple(indices[x] for x in range(y, len(indices), 3)) for y in range(0, 3)
        ]

        # Diagonal combinations
        winning_combinations.append(tuple(x for x in range(0, len(indices), 4)))
        winning_combinations.append(tuple(x for x in range(2, len(indices) - 1, 2)))

        self.winning_combinations = winning_combinations

    # returns:
    # -1 for no winner
    # 1 -- agent 0 wins
    # 2 -- agent 1 wins
    def check_for_winner(self):
        winner = -1
        
        for combination in self.winning_combinations:
            states = []
            for index in combination:
                states.append(self.squares[index])
            if all(x == 1 for x in states):
                winner = 1
            if all(x == 2 for x in states):
                winner = 2
                
        return winner

    def check_game_over(self):
        winner = self.check_for_winner()

        if winner == -1 and all(square in [1, 2] for square in self.squares):
            # tie
            return True
        elif winner in [1, 2]:
            return True
        else:
            return False

    def __str__(self):
        return str(self.squares)
    
    pass

# jogador joga primeiro
class T3Env(gym.Env):
    def __init__(self, render_mode=None):
        self.board = Tabuleiro()
        self.action_space = spaces.Discrete(9)  # 9 possíveis posições no tabuleiro
        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=int)
        self.legal_moves = self.get_mask_action()
        self.render_mode = render_mode
        
        self.first_player = 0

    def reset(self,seed=None, options=None):
        self.truncation = False
        self.terminated = False
        self.board = Tabuleiro()
        
        self.first_player = 1#random.randint(0, 2)
        
        if self.first_player != 0:
            act = int(input('jogue, jogue imediatamente '))
            #act = self._get_action_random()
            self.board.play_turn(1, act)
            self.render() if self.render_mode=='human' else None
        
        return self._get_observation(), self._get_info()
        
    
    def step(self, action):
        assert self.board.squares[action] == 0, "played illegal move"
        
        win  = 1
        lose = -1
        tie  = -0.1
        reward = 0 
        
        # Agente joga primeiro 
        self.board.play_turn(0, action)
        self.render() if self.render_mode=='human' else None
        
        if self.board.check_game_over():
            winner = self.board.check_for_winner()

            if winner == -1:
                # tie
                reward += tie
            elif winner == 1:
                # agent 0 won
                reward += win
            else:
                # agent 1 won
                reward += lose
            
            return self._get_observation(), reward, True, False, self._get_info()
        
    
        # Aleatório joga 
        act = int(input('jogue, jogue imediatamente '))
        #act = self._get_action_random()
        self.board.play_turn(1, act)
        self.render() if self.render_mode=='human' else None
        
        if self.board.check_game_over():
            winner = self.board.check_for_winner()

            if winner == -1:
                # tie
                reward += tie
            elif winner == 1:
                # agent 0 won
                reward += win
            else:
                # agent 1 won
                reward += lose
            
            return self._get_observation(), reward, True, False, self._get_info()
        
        
        return self._get_observation(), reward, False, False, self._get_info()
            
    def get_mask_action(self):
        return [i for i in range(len(self.board.squares)) if self.board.squares[i] == 0]        
    
    
    def _get_action_random(self):
        act = self.get_mask_action()
        np.random.shuffle(act)
        return act[0]
    
    def _get_observation(self):
        return np.array(self.board.squares)
        
    def _get_info(self):
        return {'info' : self.get_mask_action()}
        
    def render(self):
        k = 0
        for i in range(3):
            for j in range(3):
                if self.board.squares[k] == 0:
                    print('  | ', end='') if j!=2 else print(' ')
                elif self.board.squares[k] == 1:
                    print(' X |', end='') if j!=2 else print(' X')
                else:
                    print(' O |', end='') if j!=2 else print(' O')
                k += 1

            print('-----------') if i!= 2 else print('')
        
        return 
    
    def play_human(self, action):
        assert self.board.squares[action] == 0, "played illegal move"
        
        # Parametros de vitoria 
        win  = 1
        lose = -1
        tie  = 0
        reward = 0 
        
        # Agente joga primeiro 
        self.board.play_turn(0, action)
        self.render() if self.render_mode=='human' else None
        
        if self.board.check_game_over():
            winner = self.board.check_for_winner()

            if winner == -1:
                # tie
                reward += tie
            elif winner == 1:
                # agent 0 won
                reward += win
            else:
                # agent 1 won
                reward += lose
            
            return self._get_observation(), reward, True, False, self._get_info()
        
    
        # Humano joga 
        act = int(input('joga ai fellas '))
        self.board.play_turn(1, act)
        self.render() if self.render_mode=='human' else None
        
        if self.board.check_game_over():
            winner = self.board.check_for_winner()

            if winner == -1:
                # tie
                reward += tie
            elif winner == 1:
                # agent 0 won
                reward += win
            else:
                # agent 1 won
                reward += lose
            
            return self._get_observation(), reward, True, False, self._get_info()
        
        
        return self._get_observation(), reward, False, False, self._get_info()

    def play_human_first(self, env, action, model='MPPO_20240702-14_28_54'):
        assert self.board.squares[action] == 0, "played illegal move"
        
        # Parametros de vitoria 
        win  = 1
        lose = -1
        tie  = 0
        reward = 0
        
        # Carregando modelo 
        model = MaskablePPO.load(model)
    
        # Humano joga 
        self.board.play_turn(1, action)
        self.render() if self.render_mode=='human' else None
        
        if self.board.check_game_over():
            winner = self.board.check_for_winner()

            if winner == -1:
                # tie
                reward += tie
            elif winner == 1:
                # agent 0 won
                reward += win
            else:
                # agent 1 won
                reward += lose
            
            return self._get_observation(), reward, True, False, self._get_info()
        
        # Agente joga  
        action_masks = get_action_masks(env)
        print('action_masks', action_masks)
        
        action, _states = model.predict(self._get_observation(), action_masks=action_masks)
        print('action', action)
        
        self.board.play_turn(0, action)
        self.render() if self.render_mode=='human' else None
        
        if self.board.check_game_over():
            winner = self.board.check_for_winner()

            if winner == -1:
                # tie
                reward += tie
            elif winner == 1:
                # agent 0 won
                reward += win
            else:
                # agent 1 won
                reward += lose
            
            return self._get_observation(), reward, True, False, self._get_info()
               
        return self._get_observation(), reward, False, False, self._get_info()


if __name__ == '__main__':
    env = T3Env('human')

    for _ in range(3):
        obs, info = env.reset()
        print('first player = ', env.first_player)
        print('obs',obs)
        print('info',info)
        
        terminated = False
        
        while terminated != True:
            act = int(input('play ai fellas '))
            obs, rw, term, trun, info = env.step(act)  
            print('obs',obs)
            print('info',info)
            print('reward',rw)
            print(term)
            terminated = term
        print('==========================')
    