# cube.py
import numpy as np
import pygame
from config import *
from camera import Camera3D
import copy
import math

class Quadruped:
    def __init__(self, position, vertices=None, vectrices_dict=None, rotation = np.array([0.0, 0.0, 0.0]), velocity = np.array([0.0, 0.0, 0.0]), color = (255, 255, 255)):
        self.initial_position = position.copy()
        self.initial_velocity = velocity.copy()
        self.initial_rotation = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_angular_velocity = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_rotation = rotation.copy()
        self.initial_vertices = vertices.copy()
        self.initial_vectrices_dict = copy.deepcopy(vectrices_dict)
        self.color = color

        self.position = position # position en x, y, z du centre du quadruped
        self.velocity = velocity
        self.rotation = rotation # rotation en radians
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.vertices = vertices.copy()
        self.vectrices_dict = copy.deepcopy(vectrices_dict)

        self.rotated_vertices = self.get_vertices()
    
    def reset(self):
        self.position = self.initial_position.copy()
        self.vertices = self.initial_vertices.copy()
        self.velocity = self.initial_velocity.copy() * np.random.rand(3)
        self.rotation = self.initial_rotation.copy() * np.random.rand(3)
        self.angular_velocity = self.initial_angular_velocity.copy() * np.random.rand(3)
    
    def get_vertices(self):
        """Returns the current vertices with position and rotation applied"""
        rotated_vertices = []
        
        for part in self.vertices:
            part_vertices = []
            for vertex in part:
                # Apply rotation
                rotated_vertex = self.apply_rotation(vertex)
                # Apply position offset
                final_vertex = rotated_vertex + self.position
                part_vertices.append(final_vertex)
            rotated_vertices.append(part_vertices)
        
        return rotated_vertices
    
    def apply_rotation(self, vertex):
        """Apply rotation matrix to a vertex"""
        # Create rotation matrices for each axis
        rx = np.array([
            [1, 0, 0],
            [0, np.cos(self.rotation[0]), -np.sin(self.rotation[0])],
            [0, np.sin(self.rotation[0]), np.cos(self.rotation[0])]
        ])
        
        ry = np.array([
            [np.cos(self.rotation[1]), 0, np.sin(self.rotation[1])],
            [0, 1, 0],
            [-np.sin(self.rotation[1]), 0, np.cos(self.rotation[1])]
        ])
        
        rz = np.array([
            [np.cos(self.rotation[2]), -np.sin(self.rotation[2]), 0],
            [np.sin(self.rotation[2]), np.cos(self.rotation[2]), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations
        rotation_matrix = rz @ ry @ rx
        return rotation_matrix @ vertex

    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le quadruped 3D avec projection et profondeur"""
        self.rotated_vertices = self.get_vertices()
        
        # Draw each part of the quadruped
        for part_vertices in self.rotated_vertices:
            # Project vertices to 2D
            projected_vertices = []
            for vertex in part_vertices:
                projected = camera.project_point(vertex)
                if projected is not None:
                    projected_vertices.append(projected)
            
            # Draw the part if we have enough vertices
            if len(projected_vertices) >= 3:
                # Draw as a polygon (assuming vertices form a cube face)
                if len(projected_vertices) == 4:
                    pygame.draw.polygon(screen, self.color, projected_vertices, 1)
                else:
                    # Draw as individual points if not a complete face
                    for point in projected_vertices:
                        pygame.draw.circle(screen, self.color, (int(point[0]), int(point[1])), 2)