import numpy as np
import pygame
from config import *
from camera import Camera3D
import math

class Cube3D:
    def __init__(self, position, x_length = 1, y_length = 1, z_length = 1):
        self.initial_position = position.copy()
        self.initial_velocity = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_rotation = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_angular_velocity = np.array([0.0, 0.0, 0.0]).copy()

        self.position = position # position en x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])  # rotation en radians
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.x_length = float(x_length)
        self.y_length = float(y_length)
        self.z_length = float(z_length)
        
    def update(self):
        # Physique 3D complète
        self.velocity += GRAVITY * DT
        self.position += self.velocity * DT
        self.rotation += self.angular_velocity * DT
        
        # Collision avec le sol (y = 0)
        if self.position[1] - self.y_length / 2 < 0:
            self.position[1] = self.y_length / 2
            self.velocity[1] *= -0.7  # rebond avec perte d'énergie
            self.velocity[0] *= 0.95  # friction
            self.velocity[2] *= 0.95  # friction
            
        # Collision avec les murs
        wall_size = 10
        for i in [0, 2]:  # x et z
            if abs(self.position[i]) + self.x_length / 2 > wall_size:
                self.position[i] = np.sign(self.position[i]) * (wall_size - self.x_length / 2)
                self.velocity[i] *= -0.8
    
    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le cube 3D avec projection et profondeur"""
        # Calculer les 8 sommets du cube
        vertices = []
        for x in [-self.x_length/2, self.x_length/2]:
            for y in [-self.y_length/2, self.y_length/2]:
                for z in [-self.z_length/2, self.z_length/2]:
                    vertex = np.array([x, y, z])
                    vertices.append(vertex)
        
        # Appliquer les rotations aux sommets
        rotated_vertices = []
        for vertex in vertices:
            # Rotation autour de l'axe X (pitch)
            cos_x = math.cos(self.rotation[0])
            sin_x = math.sin(self.rotation[0])
            y1 = vertex[1] * cos_x - vertex[2] * sin_x
            z1 = vertex[1] * sin_x + vertex[2] * cos_x
            rotated_vertex = np.array([vertex[0], y1, z1])
            
            # Rotation autour de l'axe Y (yaw)
            cos_y = math.cos(self.rotation[1])
            sin_y = math.sin(self.rotation[1])
            x2 = rotated_vertex[0] * cos_y + rotated_vertex[2] * sin_y
            z2 = -rotated_vertex[0] * sin_y + rotated_vertex[2] * cos_y
            rotated_vertex = np.array([x2, rotated_vertex[1], z2])
            
            # Rotation autour de l'axe Z (roll)
            cos_z = math.cos(self.rotation[2])
            sin_z = math.sin(self.rotation[2])
            x3 = rotated_vertex[0] * cos_z - rotated_vertex[1] * sin_z
            y3 = rotated_vertex[0] * sin_z + rotated_vertex[1] * cos_z
            rotated_vertex = np.array([x3, y3, rotated_vertex[2]])
            
            # Ajouter la position du cube
            final_vertex = self.position + rotated_vertex
            rotated_vertices.append(final_vertex)
        
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in rotated_vertices:
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
                    pygame.draw.line(screen, WHITE, start, end, 2)
    
    def reset(self):
        self.position = self.initial_position.copy()
        self.velocity = self.initial_velocity.copy()
        self.rotation = self.initial_rotation.copy()
        self.angular_velocity = self.initial_angular_velocity.copy()
    
    def get_face_center(self, face_index: int) -> np.array:
        """
        Retourne le centre de la face donnée par son index parmi les 6 faces
        Prend en compte la rotation du cube
        """
        # Définir le centre de la face dans le repère local du cube
        if face_index == 0:
            local_center = np.array([0, self.y_length / 2, 0])
        elif face_index == 1:
            local_center = np.array([self.x_length / 2, 0, 0])
        elif face_index == 2:
            local_center = np.array([0, -self.y_length / 2, 0])
        elif face_index == 3:
            local_center = np.array([-self.x_length / 2, 0, 0])
        elif face_index == 4:
            local_center = np.array([0, 0, self.z_length / 2])
        elif face_index == 5:
            local_center = np.array([0, 0, -self.z_length / 2])
        else:
            raise ValueError(f"Face index must be between 0 and 5, got {face_index}")
        
        # Appliquer les rotations
        # Rotation autour de l'axe X (pitch)
        cos_x = math.cos(self.rotation[0])
        sin_x = math.sin(self.rotation[0])
        y1 = local_center[1] * cos_x - local_center[2] * sin_x
        z1 = local_center[1] * sin_x + local_center[2] * cos_x
        rotated_center = np.array([local_center[0], y1, z1])
        
        # Rotation autour de l'axe Y (yaw)
        cos_y = math.cos(self.rotation[1])
        sin_y = math.sin(self.rotation[1])
        x2 = rotated_center[0] * cos_y + rotated_center[2] * sin_y
        z2 = -rotated_center[0] * sin_y + rotated_center[2] * cos_y
        rotated_center = np.array([x2, rotated_center[1], z2])
        
        # Rotation autour de l'axe Z (roll)
        cos_z = math.cos(self.rotation[2])
        sin_z = math.sin(self.rotation[2])
        x3 = rotated_center[0] * cos_z - rotated_center[1] * sin_z
        y3 = rotated_center[0] * sin_z + rotated_center[1] * cos_z
        rotated_center = np.array([x3, y3, rotated_center[2]])
        
        # Ajouter la position du cube
        return self.position + rotated_center