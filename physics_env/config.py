# config.py
import numpy as np

# --- Configuration ---
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 800
FPS = 60

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

# --- Contact / Friction ----------------------------------------------------
MU_STATIC          = 2.0     # très forte adhérence (le quad ne doit pas glisser)
MU_DYNAMIC         = 1.0     # coulomb cinétique
STATIC_VEL_THRESH  = 0.03    # m/s : en‑dessous, on essaye d’annuler v_t