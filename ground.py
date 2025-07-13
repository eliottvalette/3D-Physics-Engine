import numpy as np
import pygame
from config import *
from camera import Camera3D

class Ground:
    def __init__(self, size=20):
        self.size = size
        self._3d_world_points = []
        
    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le sol en 3D"""
        for x in range(-self.size, self.size + 1):
            for z in range(-self.size, self.size + 1):
                # Créer un point au sol
                point_3d = np.array([x, 0, z])
                self._3d_world_points.append(point_3d)
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
        origin = np.array([0, 0, 0])
        axes = [
            (np.array([5, 0, 0]), RED),    # X
            (np.array([0, 5, 0]), GREEN),  # Y  
            (np.array([0, 0, 5]), BLUE)    # Z
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


class Staircase:
    def __init__(self, size=20, num_steps=10, step_width=1.0, step_height=1.0, step_depth=1.0, start_x=0, start_z=0):
        self.size = size
        self.num_steps = num_steps
        self.step_width = step_width
        self.step_height = step_height
        self.step_depth = step_depth
        self.start_x = start_x
        self.start_z = start_z
        self._3d_world_points = []

    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """
        Dessine la rampe en 3D
        Je veux sur un carré de 40x40, la partie droite 40x20 soit juste plate, 
        et la partie gauche 40x20 un escalier.
        """
        for x in range(-self.size, self.size + 1):  # -20 à 20 en x
            for z in range(-self.size, 1):  # -20 à 0 en z
                point_3d = np.array([x, 0, z])
                self._3d_world_points.append(point_3d)
                projected = camera.project_3d_to_2d(point_3d)
                
                if projected:
                    depth = projected[2]
                    color_intensity = max(0, min(255, 255 - depth * 10))
                    color = (color_intensity, color_intensity, color_intensity)
                    
                    size = np.clip(int(10 / depth), 1, 10)
                    rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                    if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                        rect.right > 0 and rect.bottom > 0):
                        pygame.draw.rect(screen, color, rect)
        
        # Partie gauche avec escalier
        for step in range(self.num_steps):
            step_y = step * self.step_height
            step_z_start = step * self.step_depth
            
            # Dessiner la marche horizontale
            for x in range(-self.size, self.size + 1):  # -20 à 20 en x
                for z in range(int(step_z_start), int(step_z_start + self.step_depth)): 
                    point_3d = np.array([x, step_y, z])
                    self._3d_world_points.append(point_3d)
                    projected = camera.project_3d_to_2d(point_3d)
                    
                    if projected:
                        depth = projected[2]
                        color_intensity = max(0, min(255, 255 - depth * 10))
                        color = (color_intensity, color_intensity, color_intensity)
                        
                        size = np.clip(int(10 / depth), 1, 10)
                        rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                        if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                            rect.right > 0 and rect.bottom > 0):
                            pygame.draw.rect(screen, color, rect)
            
            # Dessiner la contremarche verticale (si ce n'est pas la dernière marche)
            if step < self.num_steps - 1:
                for x in range(-self.size, self.size + 1):
                    for y in range(int(step_y), int(step_y + self.step_height)):
                        point_3d = np.array([x, y, step_z_start + self.step_depth])
                        self._3d_world_points.append(point_3d)
                        projected = camera.project_3d_to_2d(point_3d)
                        
                        if projected:
                            depth = projected[2]
                            color_intensity = max(0, min(255, 255 - depth * 10))
                            color = (color_intensity, color_intensity, color_intensity)
                            
                            size = np.clip(int(10 / depth), 1, 10)
                            rect = pygame.Rect(projected[0] - size//2, projected[1] - size//2, size, size)
                            if (0 <= rect.left < WINDOW_WIDTH and 0 <= rect.top < WINDOW_HEIGHT and
                                rect.right > 0 and rect.bottom > 0):
                                pygame.draw.rect(screen, color, rect)
    
    def draw_axes(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine les axes 3D pour référence"""
        origin = np.array([0, 0, 0])
        axes = [
            (np.array([5, 0, 0]), RED),    # X
            (np.array([0, 5, 0]), GREEN),  # Y  
            (np.array([0, 0, 5]), BLUE)    # Z
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