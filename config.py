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