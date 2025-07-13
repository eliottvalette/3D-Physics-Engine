import numpy as np
import pygame
from config import *
from camera import Camera3D
import math

class Cube3D:
    def __init__(self, position, x_length = 1, y_length = 1, z_length = 1, rotation = np.array([0.0, 0.0, 0.0]), velocity = np.array([0.0, 0.0, 0.0])):
        self.initial_position = position.copy()
        self.initial_velocity = velocity.copy()
        self.initial_rotation = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_angular_velocity = np.array([0.0, 0.0, 0.0]).copy()
        self.initial_rotation = rotation.copy()

        self.position = position # position en x, y, z
        self.velocity = velocity
        self.rotation = rotation # rotation en radians
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.x_length = float(x_length)
        self.y_length = float(y_length)
        self.z_length = float(z_length)

        self.rotated_vertices = self.get_vertices()
        
    def update_ground_only_simple(self):
        # Mettre à jour les sommets du cube
        self.rotated_vertices = self.get_vertices()

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

    def update_ground_only_complex(self):
        """
        Collision plus complexe entre le cube (les sommets pour simplifier les calculs)
        et le sol (plat à une hauteur fixe). Si un sommet passe sous le sol, on applique une force verticale
        pour l'empêcher de passer sous le sol, ce qui modifie la vitesse linéaire et angulaire du cube.
        Pas de rebond, c'est un bloc de ciment sur du ciment.
        """
        # Mettre à jour les sommets du cube
        self.rotated_vertices = self.get_vertices()

        # Constantes physiques
        mass = 1.0
        # Tenseur d'inertie d'un pavé droit homogène (diagonal)
        I = np.array([
            (1/12) * mass * (self.y_length**2 + self.z_length**2),
            (1/12) * mass * (self.x_length**2 + self.z_length**2),
            (1/12) * mass * (self.x_length**2 + self.y_length**2)
        ])

        # Appliquer la gravité au centre de masse
        self.velocity += GRAVITY * DT

        # Mise à jour de la position et de la rotation
        self.position += self.velocity * DT
        self.rotation += self.angular_velocity * DT
        
        # Recalculer les sommets après mise à jour
        self.rotated_vertices = self.get_vertices()
        
        # Trier les sommets du plus proche du sol au plus loin
        sorted_vertices = sorted(self.rotated_vertices, key=lambda v: v[1])
        # is_close_to_ground si le plus proche <= 0.03
        is_close_to_ground = sorted_vertices[0][1] <= 0.03
        # is_on_ground si au moins 3 sommets sont <= 0.03
        is_on_ground = sorted_vertices[2][1] <= 0.03

        # Si le cube est au sol, appliquer une stabilisation plus agressive
        if is_close_to_ground:
            # Réduire drastiquement les vitesses horizontales et angulaires
            self.velocity[0] *= 0.9
            self.velocity[2] *= 0.9
            self.angular_velocity *= 0.9
        
        if is_on_ground:
            # Réduire drastiquement les vitesses horizontales et angulaires
            self.velocity[0] *= 0.2
            self.velocity[2] *= 0.2
            self.angular_velocity *= 0.1  # Réduction très forte quand sur le sol

            # Réduire drastiquement la vitesse verticale
            self.velocity[1] *= 0.2

        # Pour chaque sommet, vérifier la collision avec le sol
        for vertex in self.rotated_vertices:
            if vertex[1] < 0:
                # Position du sommet par rapport au centre de masse (c'est ce qui permet de casser la symétrie de la force d'impulsion)
                relative_position = vertex - self.position
                # Vitesse du sommet (translation + rotation)
                vertex_velocity = self.velocity + np.cross(self.angular_velocity, relative_position)
                # Si le sommet descend, on annule la composante verticale
                if vertex_velocity[1] < 0:
                    # Impulsion nécessaire pour annuler la vitesse verticale
                    normal = np.array([0, 1, 0])  # normale du sol
                    relative_velocity = np.dot(vertex_velocity, normal)
                    # Calcul de l'impulsion scalaire
                    r_cross_n = np.cross(relative_position, normal)
                    denom = (1/mass) + np.dot(normal, np.cross(np.divide(r_cross_n, I, out=np.zeros_like(r_cross_n), where=I!=0), relative_position))
                    if denom == 0:
                        continue
                    scalar_impulse = -relative_velocity / denom
                    # Appliquer l'impulsion au centre de masse
                    self.velocity += (scalar_impulse * normal) / mass
                    # Appliquer l'impulsion angulaire
                    self.angular_velocity += np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
                
                # Replacer le sommet sur le sol en ajustant la position du centre de masse
                self.position[1] = max(self.position[1], -relative_position[1])

        # Optionnel : limiter la rotation pour éviter les dérives numériques
        self.rotation = np.mod(self.rotation, 2 * np.pi)
    
    def update_ground_and_wall_complex(self, floor_level: float, wall_distance: float):
        """
        Collision plus complexe entre le cube (les sommets pour simplifier les calculs)
        et le sol (plat à une hauteur fixe) et un mur (plat à une distance fixe). 
        Si un sommet passe sous le sol ou traverse le mur, on applique une force verticale ou latérale
        pour l'empêcher de passer sous le sol ou de traverser le mur, ce qui modifie la vitesse linéaire et angulaire du cube.
        Pas de rebond, c'est un bloc de ciment sur du ciment.
        """
        # Mettre à jour les sommets du cube
        self.rotated_vertices = self.get_vertices()

        # Constantes physiques
        mass = 1.0
        # Tenseur d'inertie d'un pavé droit homogène (diagonal)
        I = np.array([
            (1/12) * mass * (self.y_length**2 + self.z_length**2),
            (1/12) * mass * (self.x_length**2 + self.z_length**2),
            (1/12) * mass * (self.x_length**2 + self.y_length**2)
        ])

        # Appliquer la gravité au centre de masse
        self.velocity += GRAVITY * DT

        # Mise à jour de la position et de la rotation
        self.position += self.velocity * DT
        self.rotation += self.angular_velocity * DT
        
        # Recalculer les sommets après mise à jour
        self.rotated_vertices = self.get_vertices()
        
        # Trier les sommets du plus proche du sol au plus loin
        sorted_vertices = sorted(self.rotated_vertices, key=lambda v: v[1])
        # is_close_to_ground si le plus proche <= 0.03
        is_close_to_ground = sorted_vertices[0][1] <= 0.03
        # is_on_ground si au moins 3 sommets sont <= 0.03
        is_on_ground = sorted_vertices[2][1] <= 0.03

        # Si le cube est au sol, appliquer une stabilisation plus agressive
        if is_close_to_ground:
            # Réduire drastiquement les vitesses horizontales et angulaires
            self.velocity[0] *= 0.9
            self.velocity[2] *= 0.9
            self.angular_velocity *= 0.9
        
        if is_on_ground:
            # Réduire drastiquement les vitesses horizontales et angulaires
            self.velocity[0] *= 0.2
            self.velocity[2] *= 0.2
            self.angular_velocity *= 0.1  # Réduction très forte quand sur le sol

            # Réduire drastiquement la vitesse verticale
            self.velocity[1] *= 0.2

        # Pour chaque sommet, vérifier la collision avec le sol
        for vertex in self.rotated_vertices:
            if vertex[1] < floor_level:
                # Position du sommet par rapport au centre de masse (c'est ce qui permet de casser la symétrie de la force d'impulsion)
                relative_position = vertex - self.position
                # Vitesse du sommet (translation + rotation)
                vertex_velocity = self.velocity + np.cross(self.angular_velocity, relative_position)
                # Si le sommet descend, on annule la composante verticale
                if vertex_velocity[1] < 0:
                    # Impulsion nécessaire pour annuler la vitesse verticale
                    normal = np.array([0, 1, 0])  # normale du sol
                    relative_velocity = np.dot(vertex_velocity, normal)
                    # Calcul de l'impulsion scalaire
                    r_cross_n = np.cross(relative_position, normal)
                    denom = (1/mass) + np.dot(normal, np.cross(np.divide(r_cross_n, I, out=np.zeros_like(r_cross_n), where=I!=0), relative_position))
                    if denom == 0:
                        continue
                    scalar_impulse = -relative_velocity / denom
                    # Appliquer l'impulsion au centre de masse
                    self.velocity += (scalar_impulse * normal) / mass
                    # Appliquer l'impulsion angulaire
                    self.angular_velocity += np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
                
                # Replacer le sommet sur le sol en ajustant la position du centre de masse
                self.position[1] = max(self.position[1], floor_level - relative_position[1])

        # Pour chaque sommet, vérifier la collision avec le mur (x = wall_distance)
        for vertex in self.rotated_vertices:
            if vertex[0] > wall_distance:
                # Position du sommet par rapport au centre de masse
                relative_position = vertex - self.position
                # Vitesse du sommet (translation + rotation)
                vertex_velocity = self.velocity + np.cross(self.angular_velocity, relative_position)
                # Si le sommet avance vers le mur, on annule la composante horizontale
                if vertex_velocity[0] > 0:
                    # Impulsion nécessaire pour annuler la vitesse horizontale
                    normal = np.array([-1, 0, 0])  # normale du mur (vers l'intérieur)
                    relative_velocity = np.dot(vertex_velocity, normal)
                    # Calcul de l'impulsion scalaire
                    r_cross_n = np.cross(relative_position, normal)
                    denom = (1/mass) + np.dot(normal, np.cross(np.divide(r_cross_n, I, out=np.zeros_like(r_cross_n), where=I!=0), relative_position))
                    if denom == 0:
                        continue
                    scalar_impulse = -relative_velocity / denom
                    # Appliquer l'impulsion au centre de masse
                    self.velocity += (scalar_impulse * normal) / mass
                    # Appliquer l'impulsion angulaire
                    self.angular_velocity += np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
                
                # Replacer le sommet sur le mur en ajustant la position du centre de masse
                self.position[0] = min(self.position[0], wall_distance - relative_position[0])

        # Optionnel : limiter la rotation pour éviter les dérives numériques
        self.rotation = np.mod(self.rotation, 2 * np.pi)

    def update_on_stairs(self, stairs_coordinates_flat: dict[int, list[tuple[float, float]]], stairs_coordinates_vertical: dict[int, list[tuple[float, float]]]):
        """
        Collision avancée entre le cube (les sommets pour simplifier les calculs)
        et les marches d'escalier. Chaque marche est traitée comme un petit sol avec sa hauteur.

        stairs_coordinates : dict[int, list[tuple[float, float]]]
        Chaque dict contient:
        - y: hauteur de la marche
        - step_points: liste de 4 points (x,z) définissant la marche (rectangle parallèle au sol)
        """

        # Mettre à jour les sommets du cube
        self.rotated_vertices = self.get_vertices()

        # Constantes physiques
        mass = 1.0
        # Tenseur d'inertie d'un pavé droit homogène (diagonal)
        I = np.array([
            (1/12) * mass * (self.y_length**2 + self.z_length**2),
            (1/12) * mass * (self.x_length**2 + self.z_length**2),
            (1/12) * mass * (self.x_length**2 + self.y_length**2)
        ])

        # Appliquer la gravité au centre de masse
        self.velocity += GRAVITY * DT

        # Mise à jour de la position et de la rotation
        self.position += self.velocity * DT
        self.rotation += self.angular_velocity * DT
        
        # Recalculer les sommets après mise à jour
        self.rotated_vertices = self.get_vertices()
        
        # Calculer la hauteur du sol sous chaque sommet et trier par distance au sol
        vertex_ground_distances = []
        for vertex in self.rotated_vertices:
            ground_height = self.height_of_ground_below(vertex, stairs_coordinates_flat)
            distance_to_ground = vertex[1] - ground_height
            vertex_ground_distances.append((vertex, distance_to_ground))
        
        # Trier les sommets du plus proche du sol au plus loin
        sorted_vertices_with_distances = sorted(vertex_ground_distances, key=lambda x: x[1])
        
        # is_close_to_ground si le plus proche est proche du sol
        is_close_to_ground = sorted_vertices_with_distances[0][1] <= 0.03
        # is_on_ground si au moins 3 sommets sont proches du sol
        is_on_ground = sorted_vertices_with_distances[2][1] <= 0.03

        # Si le cube est au sol, appliquer une stabilisation plus agressive
        if is_close_to_ground:
            # Réduire drastiquement les vitesses horizontales et angulaires
            self.velocity[0] *= 0.95
            self.velocity[2] *= 0.95
            self.angular_velocity *= 0.98  # Réduction plus forte de la rotation

        # Limiter la vitesse angulaire pour éviter les rotations excessives
        max_angular_velocity = 5.0  # Limite en rad/s
        self.angular_velocity = np.clip(self.angular_velocity, -max_angular_velocity, max_angular_velocity)

        # Pour chaque sommet, vérifier la collision avec les marches
        collision_detected = False  # Éviter les impulsions multiples
        for vertex in self.rotated_vertices:
            ground_height = self.height_of_ground_below(vertex, stairs_coordinates_flat)

            # Collision avec la marche
            if vertex[1] < ground_height and not collision_detected:
                collision_detected = True  # Une seule collision par frame
                
                # Position du sommet par rapport au centre de masse
                relative_position = vertex - self.position
                # Vitesse du sommet (translation + rotation)
                vertex_velocity = self.velocity + np.cross(self.angular_velocity, relative_position)
                # Si le sommet descend, on annule la composante verticale
                if vertex_velocity[1] < 0:
                    # Impulsion nécessaire pour annuler la vitesse verticale
                    normal = np.array([0, 1, 0])  # normale du sol
                    relative_velocity = np.dot(vertex_velocity, normal)
                    # Calcul de l'impulsion scalaire
                    r_cross_n = np.cross(relative_position, normal)
                    denom = (1/mass) + np.dot(normal, np.cross(np.divide(r_cross_n, I, out=np.zeros_like(r_cross_n), where=I!=0), relative_position))
                    if denom == 0:
                        continue
                    scalar_impulse = -relative_velocity / denom
                    # Appliquer l'impulsion au centre de masse
                    self.velocity += (scalar_impulse * normal) / mass
                    # Appliquer l'impulsion angulaire (réduite pour éviter l'instabilité)
                    angular_impulse = np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
                    self.angular_velocity += angular_impulse * 0.5  # Réduire l'effet de l'impulsion angulaire
                
                # Replacer le sommet sur la marche en ajustant la position du centre de masse
                # Calculer la nouvelle position Y du centre de masse pour que le sommet soit sur la marche
                new_y = ground_height - relative_position[1]
                self.position[1] = max(self.position[1], new_y)

        # Optionnel : limiter la rotation pour éviter les dérives numériques
        self.rotation = np.mod(self.rotation, 2 * np.pi)

    def height_of_ground_below(self, vertex: np.array, stairs_coordinates: dict[int, list[tuple[float, float]]]) -> float:
        """
        Retourne la hauteur du sol en dessous du sommet

        On regarde sa position (x,y), et à partir de stairs_coordinates, on cherche la hauteur du sol en dessous.
        Pour cela on regarde dans quel rectangle se trouve le sommet, et on retourne la hauteur de ce rectangle.
        """

        x, _, z = vertex

        for step_y, step_points in stairs_coordinates.items():
            if self._point_in_rectangle_x_z(x, z, step_points):
                return step_y
        
        # Si aucun rectangle trouvé, retourner 0 (sol par défaut)
        return -float('inf')
                
    def _point_in_rectangle_x_z(self, x: float, z: float, rectangle_points: list[tuple[float, float]]) -> bool:
        """
        Vérifie si un point (x, z) est dans un rectangle défini par ses 4 points
        Cette technique avec plusiers ifs permet de réduire le compute si dès le 
        premier if on peut déjà exclure le point.
        
        Les points sont définis dans l'ordre: [(x_max, z_min), (x_max, z_max), (x_min, z_max), (x_min, z_min)]
        """
        # Extraire les bornes x et z du rectangle
        x_coords = [point[0] for point in rectangle_points]
        z_coords = [point[1] for point in rectangle_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        z_min, z_max = min(z_coords), max(z_coords)
        
        # Vérifier si le point est dans le rectangle
        if x < x_min or x > x_max:
            return False
        if z < z_min or z > z_max:
            return False
        return True

    def reset(self):
        self.position = self.initial_position.copy()
        self.velocity = self.initial_velocity.copy()
        self.rotation = self.initial_rotation.copy()
        self.angular_velocity = self.initial_angular_velocity.copy()
        self.rotation = self.initial_rotation.copy()
    
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

    def get_large_bounding_box(self, camera: Camera3D):
        """
        Retourne le grand pavé droit englobant le cube a partir des sommets du cube

        Ce pavé doit aura une rotation nulle dans le plan orthonormé x, y, z
        et une position qui correspond au centre du cube
        """
        # Calculer la bounding box 3D
        x_min = min(vertex[0] for vertex in self.rotated_vertices)
        y_min = min(vertex[1] for vertex in self.rotated_vertices)
        z_min = min(vertex[2] for vertex in self.rotated_vertices)
        x_max = max(vertex[0] for vertex in self.rotated_vertices)
        y_max = max(vertex[1] for vertex in self.rotated_vertices)
        z_max = max(vertex[2] for vertex in self.rotated_vertices)
        
        # On détermine les 8 sommets du pavé droit
        vertices = []
        for x in [x_min, x_max]:
            for y in [y_min, y_max]:
                for z in [z_min, z_max]:
                    vertices.append(np.array([x, y, z]))
        
        # On les projette sur le plan x, z
        projected_vertices = []
        for vertex in vertices:
            projected = camera.project_3d_to_2d(vertex)
            if projected:
                projected_vertices.append(projected)
        return vertices, projected_vertices
        
    
    def get_vertices(self):
        """Retourne les 8 sommets du cube dans le repère monde"""
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
        
        return rotated_vertices

    def draw(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le cube 3D avec projection et profondeur"""        
        # Projeter tous les sommets
        projected_vertices = []
        for vertex in self.rotated_vertices:
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
        
    def draw_bounding_box(self, screen: pygame.Surface, camera: Camera3D):
        """Dessine le grand rectangle englobant le cube"""
        _, projected_large_vertices = self.get_large_bounding_box(camera)

        # Dessiner les faces du cube (simplifié - juste les arêtes)
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Face avant
            (4, 5), (5, 7), (7, 6), (6, 4),  # Face arrière
            (0, 4), (1, 5), (2, 6), (3, 7)   # Arêtes verticales
        ]
        
        # Dessiner les arêtes
        for edge in edges:
            if edge[0] < len(projected_large_vertices) and edge[1] < len(projected_large_vertices):
                start = projected_large_vertices[edge[0]][:2]
                end = projected_large_vertices[edge[1]][:2]
                # Vérifier que les coordonnées sont valides
                if (0 <= start[0] < WINDOW_WIDTH and 0 <= start[1] < WINDOW_HEIGHT and
                    0 <= end[0] < WINDOW_WIDTH and 0 <= end[1] < WINDOW_HEIGHT):
                    pygame.draw.line(screen, WHITE, start, end, 2)

        