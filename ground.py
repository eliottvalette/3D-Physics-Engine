import numpy as np
import pygame
from config import *
from camera import Camera3D

class Ground:
    def __init__(self, size=20):
        self.size = size
        
    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le sol en 3D"""
        for x in range(-self.size, self.size + 1):
            for z in range(-self.size, self.size + 1):
                # Créer un point au sol
                point_3d = np.array([x, 0, z], dtype=np.float64)
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    # Couleur basée sur la profondeur
                    depth = projected[2]
                    color_intensity = max(0, min(255, 255 - depth * 10))
                    color = (color_intensity, color_intensity, color_intensity)
                    
                    # Dessiner un petit carré
                    size = np.clip(int(10 / depth), 1, 10)
                    rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                    # Vérifier que le rectangle est dans les limites de l'écran
                    if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                        rect.right > 0 and rect.bottom > 0):
                        pygame.draw.rect(screen, color, rect) 
    
    def draw_axes(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine les axes 3D pour référence"""
        origin = np.array([0, 0, 0], dtype=np.float64)
        axes = [
            (np.array([5, 0, 0], dtype=np.float64), RED),    # X
            (np.array([0, 5, 0], dtype=np.float64), GREEN),  # Y  
            (np.array([0, 0, 5], dtype=np.float64), BLUE)    # Z
        ]
        
        for axis_end, color in axes:
            start_proj = camera.project_3d_to_2d(origin)
            end_proj = camera.project_3d_to_2d(axis_end)
            
            if start_proj and end_proj:
                start = start_proj[:2]
                end = end_proj[:2]
                # Vérifier que les coordonnées sont valides
                if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                    0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                    pygame.draw.line(screen, color, start, end, 1)