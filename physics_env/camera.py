# camera.py
import math
import numpy as np
from .config import *
class Camera3D:
    def __init__(self):
        self.position = np.array([0.0, 5.0, -10.0])
        self.rotation = np.array([0.0, 0.0, 0.0])  # pitch, yaw, roll (pitch = on leve et on baisse la tete, yaw = on tourne la tete à droite et à gauche, roll = on tourne le corps à droite et à gauche)
        self.fov = math.pi / 2  # 90 degrés
        self.near = 0.1
        
    def project_3d_to_2d(self, point_3d):
        """
        Projette un point 3D en coordonnées 2D d'écran
        Input : point_3d : [x, y, z] dans un repère 3D
        Output : [x, y, z] où x et y sont les coordonnées 2D et z est la profondeur (distance entre le point et la caméra)
        """
        # Translation relative à la caméra
        relative_pos = point_3d - self.position
        
        # Rotation de la caméra (simplifiée - seulement yaw et pitch)
        cos_yaw = math.cos(self.rotation[1])
        sin_yaw = math.sin(self.rotation[1])
        cos_pitch = math.cos(self.rotation[0])
        sin_pitch = math.sin(self.rotation[0])
        
        # Rotation Y (yaw)
        x = relative_pos[0] * cos_yaw - relative_pos[2] * sin_yaw
        y = relative_pos[1]
        z = relative_pos[0] * sin_yaw + relative_pos[2] * cos_yaw
        
        # Rotation X (pitch)
        y2 = y * cos_pitch - z * sin_pitch
        z2 = y * sin_pitch + z * cos_pitch

        if z2 <= self.near:  # Éviter la division par zéro
            return None # Si le point est derrière la caméra, on ne le projette pas
            
        scale = 400 / z2  # Facteur d'échelle
        
        # Limiter le facteur d'échelle pour éviter les coordonnées énormes
        if scale > 1000:
            return None
            
        screen_x = WINDOW_WIDTH // 2 + x * scale
        screen_y = WINDOW_HEIGHT // 2 - y2 * scale
        
        # Vérifier que les coordonnées ne sont pas infinies ou NaN
        if (math.isnan(screen_x) or math.isnan(screen_y) or 
            math.isinf(screen_x) or math.isinf(screen_y)):
            return None
            
        # Vérifier que les coordonnées sont dans des limites raisonnables
        if abs(screen_x) > 10000 or abs(screen_y) > 10000:
            return None
        
        return (int(screen_x), int(screen_y), z2)
    
    def get_depth(self, point_3d):
        """Retourne la profondeur d'un point pour le tri"""
        relative_pos = point_3d - self.position
        return math.sqrt(np.dot(relative_pos, relative_pos))
    
    # Déplacement de la caméra en fonction de sa rotation
    def go_straight(self, speed):
        self.position[2] += speed * math.cos(self.rotation[1])
        self.position[0] += speed * math.sin(self.rotation[1])
    
    def go_backward(self, speed):
        self.position[2] -= speed * math.cos(self.rotation[1])
        self.position[0] -= speed * math.sin(self.rotation[1])
    
    def go_left(self, speed):
        self.position[0] -= speed * math.cos(self.rotation[1])
        self.position[2] += speed * math.sin(self.rotation[1])
    
    def go_right(self, speed):
        self.position[0] += speed * math.cos(self.rotation[1])
        self.position[2] -= speed * math.sin(self.rotation[1])