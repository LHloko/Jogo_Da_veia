"""
Created on Fri Jun  7 18:59:30 2024

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
            if player == 'X' or player == 'x':
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
        for i in range(3):
            for j in range(3):
                if self.board[k] == 0:
                    print('  | ', end='') if j!=2 else print(' ')
                elif self.board[k] == 1:
                    print(' X |', end='') if j!=2 else print(' X')
                else:
                    print(' O |', end='') if j!=2 else print(' O')
                k += 1

            print('-----------') if i!= 2 else print('')



class Zuita_Env(AECEnv):
    metadata = {
        "name": "Zuita_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.jogo = Jogo_Velha()
        self.render_mode = render_mode

        self.agentes = ['x', 'o']
        self.agentes_possiveis = self.agentes[:]

        self.action_spaces = {i: spaces.Discrete(9) for i in self.agentes}
        self.observation_space = {
            i: spaces.Dict(
                 {
                 'observation': spaces.Box(0, 1, (3,3,2), dtype=np.int8),
                 'action_mask': spaces.Box(0, 1, (9,), dtype=np.int8),
                 }
            ) for i in self.agentes
        }

        self.rewards = {i: 0 for i in self.agentes}
        self.terminations = {i: False for i in self.agentes}
        self.truncations = {i: False for i in self.agentes}

        # Action mask
        self.infos = {i: {"movimento_valido": list(range(0, 9))} for i in self.agentes}

        # Utils do pettinzoo
        self._seletor_de_agente = agent_selector(self.agentes)
        self.agente_selecionado = self._seletor_de_agente.reset()

    def observe(self, agente):
        malha        = np.array(self.jogo.board).reshape(3,3)
        atual_ply    = self.agentes_possiveis.index(agente)
        oponente_ply = (atual_ply + 1) % 2
 
        atual_board    = np.equal(malha, atual_ply+1)
        oponente_board = np.equal(malha, oponente_ply+1)
 
        # Observaçao
        obs = np.stack([atual_board, oponente_board], axis=2).astype(np.int8)

        # Selecionador de jogador e açoes validas
        mov_valido = self._movimentos_validos() if agente == self.agente_selecionado else []

        # Action mask
        action_mask = np.zeros(9,'int8')
        for i in mov_valido:
            action_mask[i] = 1

        return {'observation': obs, 'action_mask': action_mask}

    def _movimentos_validos(self):
        board = self.jogo.board
        return [i for i in range(len(board)) if board[i] == 0 ]



    # Nao sei o que e isso necessariamente
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # Nao sei o que e isso necessariamente
    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, action):
        # avalia se o jogo nao acabou do outro lado da mesa
        if (
            self.truncations[self.agente_selecionado]
            or self.terminations[self.agente_selecionado]
            ):
            return self._was_dead_step(action)

        # Verifica se a açao eh valida mesmo, senao imprime um errao na tela
        assert self.jogo.board[action] == 0, "açao invalida, fellas"

        # jogada vis factum
        jugadoh = self.agentes.index(self.agente_selecionado)
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
            self.terminations = {i: True for i in self.agentes}

        # troca o jogador
        next_jogador = self._seletor_de_agente.next()
        self.agente_selecionado = next_jogador

        # Sistema de recompensas acumuladas
        self._cumulative_rewards[self.agente_selecionado] = 0
        self._accumulate_rewards()

        # Renderiza o ambiente
        if self.render_mode == 'human':
            self.jogo.render()

    def reset(self, seed=None, options=None):
        self.jogo = Zuita_Env()
        self.agentes = self.agentes_possiveis[:]
        self.rewards = {i: 0 for i in self.agentes}
        self._cumulative_rewards = {i: 0 for i in self.agentes}
        self.terminations = {i: False for i in self.agentes}
        self.truncations = {i: False for i in self.agentes}
        self.infos = {i: {} for i in self.agentes}

        # Seleciona o primeiro agente
        self._seletor_de_agente.reinit(self.agentes)
        self._seletor_de_agente.reset()
        self.agente_selecionado = self._seletor_de_agente.reset()

def main():
    env = Zuita_Env(render_mode='human')

    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
    
        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask)
    
        env.step(action)


    pass

if __name__ == "__main__":
    main()
