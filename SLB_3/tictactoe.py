from jogo_da_veia import zuita_game_v2 as zg

import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import pettingzoo.utils

class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper para permitir que ambientes PettingZoo sejam usados ​​com mascaramento de ação ilegal SB3."""

    def reset(self, seed=None, options=None):
        """Função de redefinição semelhante a um ginásio que atribui espaços de obs/ação iguais para cada agente.

        Isso é necessário porque o SB3 foi projetado para RL de agente único e não espera que os espaços obs/ação sejam funções
        """
        super().reset(seed, options)

        # Retire a máscara de ação do espaço de observação
        self.observation_space = super().observation_space(self.possible_agents[0])["observation"]

        self.action_space = super().action_space(self.possible_agents[0])

        # Retornar observação inicial, informações (os ambientes PettingZoo AEC não o fazem por padrão)
        return self.observe(self.agent_selection), {}

    def step(self, action):
        """Função de etapa semelhante a um ginásio, retornando observação, recompensa, rescisão, truncamento, informações."""
        super().step(action)

        return super().last()

    def observe(self, agent):
        """Retorna apenas a observação bruta, removendo a máscara de ação."""
        return super().observe(agent)["observation"]

    def action_mask(self):
        """Função separada usada para acessar a máscara de ação."""
        i = super().unwrapped.observe(self.agent_selection)['action_mask']

        return i

def mask_fn(env):
    # Faça o que quiser nesta função para retornar a máscara de ação
    # para o ambiente atual. Neste exemplo, assumimos que o ambiente tem um
    # método útil em que podemos confiar.

    return env.action_mask()

def train(env, steps=10_000, seed=None, **env_kwargs):
    # Define a funçao de treinamento do agente

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Convertendo o ambiente para multiplayer
    env = SB3ActionMaskWrapper(env)
    env.reset()

    env = ActionMasker(env, mask_fn)

    # Criando o Modelo
    model = MaskablePPO(policy          = MaskableActorCriticPolicy,
                        env             = env,
                        tensorboard_log = './maskablePPO_billybat_ts/',
                        verbose         = 1,
                        )

    # Treiando o agente
    model.learn(total_timesteps     = steps,
                progress_bar        = False,
                reset_num_timesteps = True,
                tb_log_name         = "maskablePPO_velha_billybat",
                )

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()

# =========================================================================== #
if __name__ == "__main__":
    env = zg.Zuita_Env(render_mode=None)
    train(env,10000, {})

