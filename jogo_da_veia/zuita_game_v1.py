"""
Created on Thu Jun 13 20:09:24 2024

@author: lbalieiro@lince.lab
"""

# Criar o ambiente com o petinzoo

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.test import parallel_api_test

import numpy as np

from gymnasium import spaces

'''  #  '''
#  X = 1  #
#  O = 2  #
'''  #  '''

class Jogo_Velha:
    # Definir a malha do jogo
    def __init__(self):
        # # # # #
        # 0 1 2 #
        # 3 4 5 #
        # 6 7 8 #
        # # # # #
        self.board = [0] * 9

    # Funçao para por peças
    def playar(self, player, pos):
        if self.board[pos] != 0:
            return False
        else:
            if player == 'X' or player == 'x' or player == 1:
                self.board[pos] = 1
            elif player == 'o' or player == 'O' or player == 0:
                self.board[pos] = 2

        return True

    # Funçao que checa se o jogo acabou ou deu velha
    def check_vitoria(self):
        matriz = np.array(self.board).reshape(3, 3)

        # Empate
        if np.all(matriz):
            return 0

        winner = -1 # Jogo nao acabou
        for i in range(3):
            # Verificar Linhas e colunas
            if np.all(matriz[i, :] == 1) or np.all(matriz[:, i] == 1):
                winner = 1
            if np.all(matriz[i, :] == 2) or np.all(matriz[:, i] == 2):
                winner = 2

            # Verificar Diagonais
            if np.all(np.diag(matriz) == 1) or np.all(np.diag(np.fliplr(matriz)) == 1):
                winner = 1
            if np.all(np.diag(matriz) == 2) or np.all(np.diag(np.fliplr(matriz)) == 2):
                winner = 2

        return winner

    # Funçao que imprime o jogo
    def render(self):
        k = 0
        print(self.board)
        for i in range(3):
            for j in range(3):
                if self.board[k] == 0:
                    print('   |', end='') if j!=2 else print('  ')
                elif self.board[k] == 1:
                    print(' X |', end='') if j!=2 else print(' X')
                else:
                    print(' O |', end='') if j!=2 else print(' O')
                k += 1

            print('-----------') if i!= 2 else print('')


# =========================================================================== #

class Zuita_Env(AECEnv):
    metadata = {
        "name": "Zuita_v1",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.jogo = Jogo_Velha()
        self.render_mode = render_mode

        # Agentes descritos na classe do jogo
        self.agents = [1,0]
        self.possible_agents = self.agents[:]

        # Espaços para o ambiente
        self.action_spaces = {i: spaces.Discrete(9) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                 {
                 'observation': spaces.Box(0, 1, (3,3,2), dtype=np.int8),
                 'action_mask': spaces.Box(0, 1, (9,), dtype=np.int8),
                 }
            ) for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}

        # Action mask
        self.infos = {i: {"movimento_valido": list(range(0, 9))} for i in self.agents}

        # Utils do pettinzoo
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agente):
        malha        = np.array(self.jogo.board).reshape(3,3)
        atual_ply    = self.possible_agents.index(agente)
        oponente_ply = (atual_ply + 1) % 2
 
        atual_board    = np.equal(malha, atual_ply+1)
        oponente_board = np.equal(malha, oponente_ply+1)
 
        # Observaçao
        obs = np.stack([atual_board, oponente_board], axis=2).astype(np.int8)

        # Selecionador de jogador e açoes validas
        mov_valido = self._movimentos_validos() if agente == self.agent_selection else []

        # Action mask
        action_mask = np.zeros(9,'int8')
        for i in mov_valido:
            action_mask[i] = 1

        return {'observation': obs, 'action_mask': action_mask}

    def _movimentos_validos(self):
        board = self.jogo.board
        return [i for i in range(len(board)) if board[i] == 0 ]


    # Funçao que retorna a observacao do agente
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # Funçao que retorna a action do agente
    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, action):
        # avalia se o jogo nao acabou do outro lado da mesa
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
            ):
            return self._was_dead_step(action)

        # Verifica se a açao eh valida mesmo, senao imprime um errao na tela
        assert self.jogo.board[action] == 0, "açao invalida, fellas"

        # jogada vis factum
        jugadoh = self.agents.index(self.agent_selection)
        self.jogo.playar(jugadoh, action)


        # FLUXO DE GAMEPLAY
        winner = self.jogo.check_vitoria()
        if winner != -1:
            if winner == 0: # Empate
                pass
            elif winner == 1:   # X ganhou
                self.rewards[self.agents[0]] += 1
                self.rewards[self.agents[1]] -= 1
            else:               # O ganhou
                self.rewards[self.agents[0]] += 1
                self.rewards[self.agents[1]] -= 1

            # Caso o jogo empate ambos recebem truncado e sem recompensa
            self.terminations = {i: True for i in self.agents}

        # troca o jogador
        next_jogador = self._agent_selector.next()
        self.agent_selection = next_jogador

        # Sistema de recompensas acumuladas
        self._cumulative_rewards[self.agent_selection] = 0
        self._accumulate_rewards()

        # Renderiza o ambiente
        if self.render_mode == 'human':
            self.jogo.render()

    def reset(self, seed=None, options=None):

        self.jogo = Zuita_Env()
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        # Seleciona o primeiro agente
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

# =========================================================================== #

if __name__ == "__main__":
    env = Zuita_Env(render_mode='human')
    env.reset()
    print(env.last())







