# main.py
from train import main_training_loop
from physics_env.config import set_seed, EPISODES, GAMMA, ALPHA, STATE_SIZE, ACTION_SIZE, RENDERING
from agent import QuadrupedAgent
from physics_env.quadruped_env import QuadrupedEnv
import torch
from collections import deque
import gc


# Définir la graine pour la reproductibilité
set_seed(43)

# Définir le device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
device = 'cpu'  # Forcer l'utilisation du CPU

agent = QuadrupedAgent(
    state_size=STATE_SIZE,
    device=device,
    action_size=ACTION_SIZE,
    gamma=GAMMA,
    learning_rate=ALPHA,
    load_model=False,
    load_path=f"saved_models/quadruped_agent.pth",
)

# Démarrer l'entraînement
main_training_loop(agent, episodes=EPISODES, rendering=RENDERING, render_every=25)