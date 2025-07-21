# quadruped.py
import numpy as np
import pygame
# from config import *
from .camera import Camera3D
import copy
import math

WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 800

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
        
        # Get shoulder and elbow positions from vectrices_dict if available
        self.shoulder_positions = vectrices_dict.get('shoulder_positions', []) if vectrices_dict else []
        self.elbow_positions = vectrices_dict.get('elbow_positions', []) if vectrices_dict else []

        # Articulations (épaules et coudes) - angles en radians
        # Épaules: 0=Front Right, 1=Front Left, 2=Back Right, 3=Back Left
        self.shoulder_angles = np.array([0.0, 0.0, 0.0, 0.0])
        # Coudes: 0=Front Right, 1=Front Left, 2=Back Right, 3=Back Left
        self.elbow_angles = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Angles initiaux
        self.initial_shoulder_angles = self.shoulder_angles.copy()
        self.initial_elbow_angles = self.elbow_angles.copy()

        self.rotated_vertices = self.get_vertices()

        self.prev_vertices = None
    
    def reset_random(self):
        self.position = self.initial_position.copy()
        self.vertices = self.initial_vertices.copy()
        self.velocity = self.initial_velocity.copy() + np.random.rand(3)
        self.rotation = self.initial_rotation.copy() + np.random.rand(3)
        self.angular_velocity = self.initial_angular_velocity.copy() + np.random.rand(3)
        self.shoulder_angles = self.initial_shoulder_angles.copy()
        self.elbow_angles = self.initial_elbow_angles.copy()
        self.rotated_vertices = self.get_vertices()
    
    def reset(self):
        self.position = self.initial_position.copy()
        self.vertices = self.initial_vertices.copy()
        self.velocity = self.initial_velocity.copy()
        self.rotation = self.initial_rotation.copy()
        self.angular_velocity = self.initial_angular_velocity.copy()
        self.shoulder_angles = self.initial_shoulder_angles.copy()
        self.elbow_angles = self.initial_elbow_angles.copy()
        self.rotated_vertices = self.get_vertices()
    
    def get_vertices(self):
        """Retourne les vertices du quadruped dans le repère monde avec transformations appliquées"""
        # Appliquer les rotations aux sommets
        rotated_vertices = []
        
        # Structure des vertices: [body(8), upper_leg_0(8), upper_leg_1(8), upper_leg_2(8), upper_leg_3(8), 
        #                          lower_leg_0(8), lower_leg_1(8), lower_leg_2(8), lower_leg_3(8)]
        
        for i, vertex in enumerate(self.vertices):
            # Déterminer à quelle partie appartient ce vertex
            part_index = i // 8  # 0=body, 1-4=upper_legs, 5-8=lower_legs
            
            # Appliquer les transformations d'articulation selon la partie
            transformed_vertex = vertex.copy()
            
            if part_index == 0:  # Body - pas de transformation d'articulation
                pass
            elif 1 <= part_index <= 4:  # Upper legs - transformation d'épaule seulement
                leg_index = part_index - 1
                shoulder_angle = self.shoulder_angles[leg_index]
                
                # Utiliser la position d'épaule calculée dynamiquement
                shoulder_center = self.shoulder_positions[leg_index]
                
                # Appliquer la rotation d'épaule autour de l'axe X
                relative_pos = transformed_vertex - shoulder_center
                cos_shoulder = math.cos(shoulder_angle)
                sin_shoulder = math.sin(shoulder_angle)
                y_new = relative_pos[1] * cos_shoulder - relative_pos[2] * sin_shoulder
                z_new = relative_pos[1] * sin_shoulder + relative_pos[2] * cos_shoulder
                transformed_vertex = shoulder_center + np.array([relative_pos[0], y_new, z_new])
                
            elif 5 <= part_index <= 8:  # Lower legs - transformation d'épaule + coude
                leg_index = part_index - 5
                shoulder_angle = self.shoulder_angles[leg_index]
                elbow_angle = self.elbow_angles[leg_index]
                
                # 1. D'abord appliquer la rotation d'épaule (même que pour upper leg)
                shoulder_center = self.shoulder_positions[leg_index]
                
                # Appliquer la rotation d'épaule
                relative_pos = transformed_vertex - shoulder_center
                cos_shoulder = math.cos(shoulder_angle)
                sin_shoulder = math.sin(shoulder_angle)
                y_new = relative_pos[1] * cos_shoulder - relative_pos[2] * sin_shoulder
                z_new = relative_pos[1] * sin_shoulder + relative_pos[2] * cos_shoulder
                transformed_vertex = shoulder_center + np.array([relative_pos[0], y_new, z_new])
                
                # 2. Ensuite appliquer la rotation de coude (par rapport à la position après épaule)
                # Le point de coude doit aussi être transformé par la rotation d'épaule
                elbow_center_original = self.elbow_positions[leg_index]
                
                # Appliquer la même transformation d'épaule au point de coude
                elbow_relative_pos = elbow_center_original - shoulder_center
                elbow_y_new = elbow_relative_pos[1] * cos_shoulder - elbow_relative_pos[2] * sin_shoulder
                elbow_z_new = elbow_relative_pos[1] * sin_shoulder + elbow_relative_pos[2] * cos_shoulder
                elbow_center_transformed = shoulder_center + np.array([elbow_relative_pos[0], elbow_y_new, elbow_z_new])
                
                # Appliquer la rotation de coude autour du point transformé
                relative_pos = transformed_vertex - elbow_center_transformed
                cos_elbow = math.cos(elbow_angle)
                sin_elbow = math.sin(elbow_angle)
                y_new = relative_pos[1] * cos_elbow - relative_pos[2] * sin_elbow
                z_new = relative_pos[1] * sin_elbow + relative_pos[2] * cos_elbow
                transformed_vertex = elbow_center_transformed + np.array([relative_pos[0], y_new, z_new])
            
            # Appliquer les rotations globales du quadruped
            # Rotation autour de l'axe X (pitch)
            cos_x = math.cos(self.rotation[0])
            sin_x = math.sin(self.rotation[0])
            y1 = transformed_vertex[1] * cos_x - transformed_vertex[2] * sin_x
            z1 = transformed_vertex[1] * sin_x + transformed_vertex[2] * cos_x
            rotated_vertex = np.array([transformed_vertex[0], y1, z1])
            
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
            
            # Ajouter la position du quadruped
            final_vertex = self.position + rotated_vertex
            rotated_vertices.append(final_vertex)
        
        return rotated_vertices
    
    def get_vectrices_dict(self):
        return self.vectrices_dict

    def set_shoulder_angle(self, leg_index, angle):
        """Définit l'angle de l'épaule pour une patte donnée (0-3)"""
        self.shoulder_angles[leg_index] = angle
        self.rotated_vertices = self.get_vertices()
    
    def set_elbow_angle(self, leg_index, angle):
        """Définit l'angle du coude pour une patte donnée (0-3)"""
        self.elbow_angles[leg_index] = angle
        self.rotated_vertices = self.get_vertices()
    
    def adjust_shoulder_angle(self, leg_index, delta_angle):
        """Ajuste l'angle de l'épaule pour une patte donnée"""
        self.shoulder_angles[leg_index] += delta_angle
        self.rotated_vertices = self.get_vertices()
    
    def adjust_elbow_angle(self, leg_index, delta_angle):
        """Ajuste l'angle du coude pour une patte donnée"""
        self.elbow_angles[leg_index] += delta_angle
        self.rotated_vertices = self.get_vertices()
    
    def get_state(self):
        """Get the current state of the quadruped."""
        vectrices = self.get_vertices()
        velocity = self.velocity
        rotation = self.rotation
        shoulder_angles = self.shoulder_angles
        elbow_angles = self.elbow_angles
        state = np.concatenate([vectrices, velocity, rotation, shoulder_angles, elbow_angles])
        return state
    
    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le quadruped 3D avec projection et profondeur (arêtes seulement)"""        
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in self.rotated_vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:  # projected peut etre None si le point est derrière la caméra
                projected_vertices.append(projected)
        
        if len(projected_vertices) < 8:
            return  # Le quadruped est partiellement hors champ de vision, on ne dessine pas
        
        # Définir les couleurs pour chaque partie
        colors = [
            (255, 255, 255),  # Body (white)
            (0, 0, 255),      # Upper leg 0 - Front right (blue)
            (255, 0, 0),      # Upper leg 1 - Front left (red)
            (0, 255, 0),      # Upper leg 2 - Back right (green)
            (255, 255, 255),  # Upper leg 3 - Back left (white)
            (0, 0, 255),      # Lower leg 0 - Front right (blue)
            (255, 0, 0),      # Lower leg 1 - Front left (red)
            (0, 255, 0),      # Lower leg 2 - Back right (green)
            (255, 255, 255)   # Lower leg 3 - Back left (white)
        ]
        
        # Dessiner chaque partie du quadruped (body + 8 legs)
        # Chaque partie a 8 vertices, donc on dessine par groupes de 8
        parts_per_leg = 8
        total_parts = len(projected_vertices) // parts_per_leg
        
        for part_idx in range(total_parts):
            start_idx = part_idx * parts_per_leg
            end_idx = start_idx + parts_per_leg
            
            if end_idx <= len(projected_vertices):
                part_vertices = projected_vertices[start_idx:end_idx]
                part_color = colors[part_idx] if part_idx < len(colors) else self.color
                
                # Définir les arêtes pour chaque partie (cube)
                edges = [
                    (0, 1), (1, 3), (3, 2), (2, 0),  # Face avant
                    (4, 5), (5, 7), (7, 6), (6, 4),  # Face arrière
                    (0, 4), (1, 5), (2, 6), (3, 7)   # Arêtes verticales
                ]
                
                # Dessiner les arêtes de cette partie
                for edge in edges:
                    if edge[0] < len(part_vertices) and edge[1] < len(part_vertices):
                        start = part_vertices[edge[0]][:2]
                        end = part_vertices[edge[1]][:2]
                        
                        # Vérifier que les coordonnées sont valides
                        if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                            0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                            pygame.draw.line(screen, part_color, start, end, 2)
    
    def draw_premium(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le quadruped 3D avec faces pleines et dégradé de gris basé sur la profondeur"""
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in self.rotated_vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:  # projected peut etre None si le point est derrière la caméra
                projected_vertices.append(projected)
        
        if len(projected_vertices) < 8:
            return  # Le quadruped est partiellement hors champ de vision, on ne dessine pas
        
        # Définir les faces pour chaque cube (6 faces par cube)
        # Chaque face est définie par 4 indices de vertices
        cube_faces = [
            [0, 1, 3, 2],  # Face avant (bottom)
            [4, 5, 7, 6],  # Face arrière (top)
            [0, 1, 5, 4],  # Face gauche
            [2, 3, 7, 6],  # Face droite
            [0, 2, 6, 4],  # Face inférieure
            [1, 3, 7, 5]   # Face supérieure
        ]
        
        # Dessiner chaque partie du quadruped (body + 8 legs)
        # Chaque partie a 8 vertices, donc on dessine par groupes de 8
        parts_per_leg = 8
        total_parts = len(projected_vertices) // parts_per_leg
        
        # Trier les faces par profondeur pour le rendu correct (painter's algorithm)
        all_faces = []
        
        for part_idx in range(total_parts):
            start_idx = part_idx * parts_per_leg
            end_idx = start_idx + parts_per_leg
            
            if end_idx <= len(projected_vertices):
                part_vertices = projected_vertices[start_idx:end_idx]
                
                # Calculer le centre de profondeur de cette partie pour le tri
                part_depth = sum(v[2] for v in part_vertices) / len(part_vertices)
                
                # Créer les faces pour cette partie
                for face_indices in cube_faces:
                    face_vertices = [part_vertices[i] for i in face_indices if i < len(part_vertices)]
                    
                    if len(face_vertices) == 4:
                        # Calculer la profondeur moyenne de la face
                        face_depth = sum(v[2] for v in face_vertices) / len(face_vertices)
                        
                        # Calculer la normale de la face pour déterminer si elle est visible
                        # Simplification : on dessine toutes les faces pour l'instant
                        
                        all_faces.append({
                            'vertices': face_vertices,
                            'depth': face_depth,
                            'part_idx': part_idx
                        })
        
        # Trier les faces par profondeur (les plus éloignées en premier)
        all_faces.sort(key=lambda face: face['depth'], reverse=True)
        
        # Dessiner les faces triées
        for face in all_faces:
            face_vertices = face['vertices']
            depth = face['depth']
            part_idx = face['part_idx']
            
            # Calculer la couleur basée sur la profondeur (dégradé de gris)
            # Plus la profondeur est grande, plus la couleur est sombre
            base_intensity = 200  # Gris clair de base
            depth_factor = max(0, min(1, depth / 50))  # Normaliser la profondeur
            intensity = int(base_intensity * (1 - depth_factor * 0.7))  # Réduire jusqu'à 30% de l'intensité
            
            color = (intensity, intensity, intensity)
            
            # Convertir les vertices en points 2D pour pygame
            points_2d = [(int(v[0]), int(v[1])) for v in face_vertices]
            
            # Vérifier que tous les points sont dans les limites de l'écran
            valid_points = True
            for x, y in points_2d:
                if not (0 <= x < WINDOW_WIDTH and 0 <= y < WINDOW_HEIGHT):
                    valid_points = False
                    break
            
            if valid_points and len(points_2d) >= 3:
                # Dessiner la face pleine
                pygame.draw.polygon(screen, color, points_2d)
                
                # Optionnel : dessiner les contours des faces pour plus de définition
                pygame.draw.polygon(screen, (50, 50, 50), points_2d, 1)