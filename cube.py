import numpy as np
import pygame
from config import *

class Cube3D:
    def __init__(self, size, position):
        self.size = float(size)
        self.position = position # position en x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # rotation en radians
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        
    def update(self):
        # Physique 3D complète
        self.velocity += GRAVITY * DT
        self.position += self.velocity * DT
        self.rotation += self.angular_velocity * DT
        
        # Collision avec le sol (y = 0)
        if self.position[1] - self.size / 2 < 0:
            self.position[1] = self.size / 2
            self.velocity[1] *= -0.7  # rebond avec perte d'énergie
            self.velocity[0] *= 0.95  # friction
            self.velocity[2] *= 0.95  # friction
            
        # Collision avec les murs
        wall_size = 10
        for i in [0, 2]:  # x et z
            if abs(self.position[i]) + self.size / 2 > wall_size:
                self.position[i] = np.sign(self.position[i]) * (wall_size - self.size / 2)
                self.velocity[i] *= -0.8
    
    def draw(self, screen: pygame.Surface, camera):
        """Dessine le cube 3D avec projection et profondeur"""
        # Calculer les 8 sommets du cube
        half_size = self.size / 2
        vertices = []
        for x in [-half_size, half_size]:
            for y in [-half_size, half_size]:
                for z in [-half_size, half_size]:
                    vertex = self.position + np.array([x, y, z], dtype=np.float64)
                    vertices.append(vertex)
        
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:  # projected peut etre None si le point est derrière la caméra
                projected_vertices.append(projected)
        
        if len(projected_vertices) < 8:
            return  # Le cube est partiellement hors champ de vision, on ne dessine pas les faces
        
        # Dessiner les faces du cube (simplifié - juste les arêtes)
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Face avant
            (4, 5), (5, 7), (7, 6), (6, 4),  # Face arrière
            (0, 4), (1, 5), (2, 6), (3, 7)   # Arêtes verticales
        ]
        
        # Dessiner les arêtes
        for edge in edges:
            if edge[0] < len(projected_vertices) and edge[1] < len(projected_vertices):
                start = projected_vertices[edge[0]][:2]
                end = projected_vertices[edge[1]][:2]
                # Vérifier que les coordonnées sont valides
                if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                    0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                    pygame.draw.line(screen, RED, start, end, 2)