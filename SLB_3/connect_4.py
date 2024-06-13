
import glob
import os
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

import pettingzoo.utils
from pettingzoo.classic import connect_four_v3


class SB3ActionMaskWrapper(pettingzoo.utils.BaseWrapper):
    """Wrapper para permitir que ambientes PettingZoo sejam usados ​​com mascaramento de ação ilegal SB3."""

    def reset(self, seed=None, options=None):
        """Função de redefinição semelhante a um ginásio que atribui espaços de obs/ação iguais para cada agente.

        Isso é necessário porque o SB3 foi projetado para RL de agente único e não espera que os espaços obs/ação sejam funções
        """
        super().reset(seed, options)

        # Retire a máscara de ação do espaço de observação
        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
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


# =========================================================================== #


def mask_fn(env):
    # Faça o que quiser nesta função para retornar a máscara de ação
    # para o ambiente atual. Neste exemplo, assumimos que o ambiente tem um
    # método útil em que podemos confiar.

    return env.action_mask()


# =========================================================================== #


def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):

    """Treine um único modelo para jogar como cada agente em um ambiente de jogo de soma zero usando máscara de ação inválida."""
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Wrapper personalizado para converter ambientes PettingZoo para funcionar com mascaramento de ação SB3
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Deve chamar reset() para redefinir os espaços

    env = ActionMasker(env, mask_fn)  # Wrap para ativar o mascaramento (função SB3)
    # MaskablePPO se comporta da mesma forma que o PPO do SB3, a menos que o ambiente seja empacotado
    # com ActionMasker. Se o wrapper for detectado, as máscaras são automaticamente
    # recuperado e usado durante o aprendizado. Observe que MaskablePPO não aceita
    # um novo action_mask_fn kwarg, como fez em um rascunho anterior.
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=0)
    model.set_random_seed(seed)
    model.learn(total_timesteps=steps)

    model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()

# =========================================================================== #

def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):

    # Avaliar um agente treinado versus um agente aleatório
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[1]}."
    )

    try:
        latest_policy = max(
            glob.glob(f"{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)

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
                # Se houver vencedor, acompanhe, caso contrário não altere a pontuação (empate)
                if (
                    env.rewards[env.possible_agents[0]]
                    != env.rewards[env.possible_agents[1]]
                ):
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[
                        winner
                    ]  # rastreia apenas a maior recompensa (vencedor do jogo)
                # Acompanhe também recompensas negativas e positivas (penaliza movimentos ilegais)
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample(action_mask)
                else:
                    # Nota: PettingZoo espera ações inteiras # TODO: change chess to cast actions to type int?
                    act = int(
                        model.predict(
                            observation, action_masks=action_mask, deterministic=True
                        )[0]
                    )
            env.step(act)
    env.close()

    # Evite dividir por zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores

# =========================================================================== #


if __name__ == "__main__":
    env_fn = connect_four_v3

    env_kwargs = {}

    # Evaluation/training hyperparameter notes:
    # 10k steps: Winrate:  0.76, loss order of 1e-03
    # 20k steps: Winrate:  0.86, loss order of 1e-04
    # 40k steps: Winrate:  0.86, loss order of 7e-06

    # Train a model against itself (takes ~20 seconds on a laptop CPU)
    train_action_mask(env_fn, steps=100000, seed=0, **env_kwargs)

    # Evaluate 100 games against a random agent (winrate should be ~80%)
    eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs)

    # Watch two games vs a random agent
    eval_action_mask(env_fn, num_games=10, render_mode=None, **env_kwargs)
