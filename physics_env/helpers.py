# helpers.py
import numpy as np

def limit_vector(vec, max_val):
    """
    Limite un vecteur de manière douce en préservant sa direction
    """
    norm = np.linalg.norm(vec)
    if norm <= max_val:
        return vec
    else:
        return vec * (max_val / norm)