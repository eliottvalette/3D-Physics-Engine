# config.py
import numpy as np
import random as rd
import torch

# --- Configuration ---
FPS = 60
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 800

# --- Couleurs ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)

# --- Physique ---
GRAVITY = np.array([0, -9.81, 0])
DT = 1/FPS

# --- Constantes de collision ---
RESTITUTION = 0.2    # Coefficient de restitution (0 = inélastique, 1 = parfaitement élastique)
FRICTION = 0.5       # Coefficient de friction cinétique
CONTACT_THRESHOLD_BASE = 0.05  # Seuil de base pour détecter le contact avec le sol
CONTACT_THRESHOLD_MULTIPLIER = 1.5  # Multiplicateur pour le seuil dynamique

# --- Limites de vitesse ---
MAX_VELOCITY = 10.0
MAX_ANGULAR_VELOCITY = 5.0
MAX_IMPULSE = 5.0
MAX_AVERAGE_IMPULSE = 2.0

# --- Angles d'articulations ---
SHOULDER_DELTA = 0.05
ELBOW_DELTA = 0.05

# --- Contact / Friction ----------------------------------------------------
SLIP_THRESHOLD = 0.05      # le pied reste « collé » tant que |v_t| < SLIP_THRESHOLD cm/s
STATIC_FRICTION_CAP  = 50.0     # impulsion maximale transmise au quadruped

# ----- Debug Physics Simulation --------------------------------------
DEBUG_CONTACT = False       

# ----- Debug RL Training --------------------------------------
DEBUG_RL_TRAIN = True
DEBUG_RL_MODEL = True
DEBUG_RL_AGENT = True
DEBUG_RL_VIZ = True

# ----- RL Training Config --------------------------------------
EPISODES = 1000
MAX_STEPS = 1000

START_EPS = 1.0
EPS_DECAY = 0.995
EPS_MIN = 0.01

PLOT_INTERVAL = 10
SAVE_INTERVAL = 100

GAMMA = 0.99
ALPHA = 0.001
STATE_SIZE = 24
ACTION_SIZE = 16

RENDERING = True


def set_seed(seed=42):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False