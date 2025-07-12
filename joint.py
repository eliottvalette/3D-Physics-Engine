import pygame
import numpy as np
from cube import Cube3D
from camera import Camera3D
from config import *

class Joint:
    def __init__(self, object_1: Cube3D, object_2: Cube3D, face_1: int, face_2: int):
        """
        object_1 : Cube3D
        object_2 : Cube3D
        position_1 : np.array <- position du joint sur object_1
        position_2 : np.array <- position du joint sur object_2
        """
        self.object_1 = object_1
        self.object_2 = object_2
        self.face_1 = face_1
        self.face_2 = face_2
        self.position_1 = self.object_1.get_face_center(face_1)
        self.position_2 = self.object_2.get_face_center(face_2)
        self.position = (self.position_1 + self.position_2) / 2

    def update(self):
        """
        Update la position du joint sur object_1 et object_2

        Le joint est au milieu des deux points position_1 et position_2
        """
        self.position_1 = self.object_1.get_face_center(self.face_1)
        self.position_2 = self.object_2.get_face_center(self.face_2)
        self.position = (self.position_1 + self.position_2) / 2
    
    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """
        Dessine le joint sur l'écran
        Le joint est un carré de 10x10 pixels
        et on trace les lignes qui relient le joint à object_1 et object_2
        """
        projected_position_1 = camera.project_3d_to_2d(self.position_1)
        projected_position_2 = camera.project_3d_to_2d(self.position_2)
        projected_position = camera.project_3d_to_2d(self.position)

        # Dessiner le joint (carré de 10x10 pixels)
        if projected_position:
            joint_size = 4
            joint_rect = pygame.Rect(
                projected_position[0] - joint_size // 2,
                projected_position[1] - joint_size // 2,
                joint_size,
                joint_size
            )
            pygame.draw.rect(screen, WHITE, joint_rect)
            
            # Dessiner les lignes qui relient le joint aux objets
            if projected_position_1:
                pygame.draw.line(screen, GREEN, 
                               (projected_position[0], projected_position[1]),
                               (projected_position_1[0], projected_position_1[1]), 2)
            
            if projected_position_2:
                pygame.draw.line(screen, GREEN,
                               (projected_position[0], projected_position[1]),
                               (projected_position_2[0], projected_position_2[1]), 2)