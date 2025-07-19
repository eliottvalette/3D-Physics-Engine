# update_functions.py
import numpy as np
import math
import random

from config import DT, GRAVITY
from cube import Cube3D
from joint import Joint
from quadruped import Quadruped

def update_ground_only_simple(object: Cube3D):
    # Mettre à jour les sommets du cube
    object.rotated_vertices = object.get_vertices()

    # Physique 3D complète
    object.velocity += GRAVITY * DT
    object.position += object.velocity * DT
    object.rotation += object.angular_velocity * DT
    
    # Collision avec le sol (y = 0)
    if object.position[1] - object.y_length / 2 < 0:
        object.position[1] = object.y_length / 2
        object.velocity[1] *= -0.7  # rebond avec perte d'énergie
        object.velocity[0] *= 0.95  # friction
        object.velocity[2] *= 0.95  # friction
        
    # Collision avec les murs
    wall_size = 10
    for i in [0, 2]:  # x et z
        if abs(object.position[i]) + object.x_length / 2 > wall_size:
            object.position[i] = np.sign(object.position[i]) * (wall_size - object.x_length / 2)
            object.velocity[i] *= -0.8

def update_ground_only_complex(object: Cube3D):
    """
    Collision plus complexe entre le cube (les sommets pour simplifier les calculs)
    et le sol (plat à une hauteur fixe). Si un sommet passe sous le sol, on applique une force verticale
    pour l'empêcher de passer sous le sol, ce qui modifie la vitesse linéaire et angulaire du cube.
    Pas de rebond, c'est un bloc de ciment sur du ciment.
    """
    # Mettre à jour les sommets du cube
    object.rotated_vertices = object.get_vertices()

    # Constantes physiques
    mass = 1.0
    # Tenseur d'inertie d'un pavé droit homogène (diagonal)
    I = np.array([
        (1/12) * mass * (object.y_length**2 + object.z_length**2),
        (1/12) * mass * (object.x_length**2 + object.z_length**2),
        (1/12) * mass * (object.x_length**2 + object.y_length**2)
    ])

    # Appliquer la gravité au centre de masse
    object.velocity += GRAVITY * DT

    # Mise à jour de la position et de la rotation
    object.position += object.velocity * DT
    object.rotation += object.angular_velocity * DT
    
    # Recalculer les sommets après mise à jour
    object.rotated_vertices = object.get_vertices()
    
    # Trier les sommets du plus proche du sol au plus loin
    sorted_vertices = sorted(object.rotated_vertices, key=lambda v: v[1])
    # is_close_to_ground si le plus proche <= 0.03
    is_close_to_ground = sorted_vertices[0][1] <= 0.03
    # is_on_ground si au moins 3 sommets sont <= 0.03
    is_on_ground = sorted_vertices[2][1] <= 0.03

    # Si le cube est au sol, appliquer une stabilisation plus agressive
    if is_close_to_ground:
        # Réduire drastiquement les vitesses horizontales et angulaires
        object.velocity[0] *= 0.9
        object.velocity[2] *= 0.9
        object.angular_velocity *= 0.9
    
    if is_on_ground:
        # Réduire drastiquement les vitesses horizontales et angulaires
        object.velocity[0] *= 0.2
        object.velocity[2] *= 0.2
        object.angular_velocity *= 0.1  # Réduction très forte quand sur le sol

        # Réduire drastiquement la vitesse verticale
        object.velocity[1] *= 0.2

    # Pour chaque sommet, vérifier la collision avec le sol
    for vertex in object.rotated_vertices:
        if vertex[1] < 0:
            # Position du sommet par rapport au centre de masse (c'est ce qui permet de casser la symétrie de la force d'impulsion)
            relative_position = vertex - object.position
            # Vitesse du sommet (translation + rotation)
            vertex_velocity = object.velocity + np.cross(object.angular_velocity, relative_position)
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
                object.velocity += (scalar_impulse * normal) / mass
                # Appliquer l'impulsion angulaire
                object.angular_velocity += np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
            
            # Replacer le sommet sur le sol en ajustant la position du centre de masse
            object.position[1] = max(object.position[1], -relative_position[1])

    # Optionnel : limiter la rotation pour éviter les dérives numériques
    object.rotation = np.mod(object.rotation, 2 * np.pi)

def update_ground_and_wall_complex(object: Cube3D, floor_level: float, wall_distance: float):
    """
    Collision plus complexe entre le cube (les sommets pour simplifier les calculs)
    et le sol (plat à une hauteur fixe) et un mur (plat à une distance fixe). 
    Si un sommet passe sous le sol ou traverse le mur, on applique une force verticale ou latérale
    pour l'empêcher de passer sous le sol ou de traverser le mur, ce qui modifie la vitesse linéaire et angulaire du cube.
    Pas de rebond, c'est un bloc de ciment sur du ciment.
    """
    # Mettre à jour les sommets du cube
    object.rotated_vertices = object.get_vertices()

    # Constantes physiques
    mass = 1.0
    # Tenseur d'inertie d'un pavé droit homogène (diagonal)
    I = np.array([
        (1/12) * mass * (object.y_length**2 + object.z_length**2),
        (1/12) * mass * (object.x_length**2 + object.z_length**2),
        (1/12) * mass * (object.x_length**2 + object.y_length**2)
    ])

    # Appliquer la gravité au centre de masse
    object.velocity += GRAVITY * DT

    # Mise à jour de la position et de la rotation
    object.position += object.velocity * DT
    object.rotation += object.angular_velocity * DT
    
    # Recalculer les sommets après mise à jour
    object.rotated_vertices = object.get_vertices()
    
    # Trier les sommets du plus proche du sol au plus loin
    sorted_vertices = sorted(object.rotated_vertices, key=lambda v: v[1])
    # is_close_to_ground si le plus proche <= 0.03
    is_close_to_ground = sorted_vertices[0][1] <= 0.03
    # is_on_ground si au moins 3 sommets sont <= 0.03
    is_on_ground = sorted_vertices[2][1] <= 0.03

    # Si le cube est au sol, appliquer une stabilisation plus agressive
    if is_close_to_ground:
        # Réduire drastiquement les vitesses horizontales et angulaires
        object.velocity[0] *= 0.9
        object.velocity[2] *= 0.9
        object.angular_velocity *= 0.9
    
    if is_on_ground:
        # Réduire drastiquement les vitesses horizontales et angulaires
        object.velocity[0] *= 0.2
        object.velocity[2] *= 0.2
        object.angular_velocity *= 0.1  # Réduction très forte quand sur le sol

        # Réduire drastiquement la vitesse verticale
        object.velocity[1] *= 0.2

    # Pour chaque sommet, vérifier la collision avec le sol
    for vertex in object.rotated_vertices:
        if vertex[1] < floor_level:
            # Position du sommet par rapport au centre de masse (c'est ce qui permet de casser la symétrie de la force d'impulsion)
            relative_position = vertex - object.position
            # Vitesse du sommet (translation + rotation)
            vertex_velocity = object.velocity + np.cross(object.angular_velocity, relative_position)
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
                object.velocity += (scalar_impulse * normal) / mass
                # Appliquer l'impulsion angulaire
                object.angular_velocity += np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
            
            # Replacer le sommet sur le sol en ajustant la position du centre de masse
            object.position[1] = max(object.position[1], floor_level - relative_position[1])

    # Pour chaque sommet, vérifier la collision avec le mur (x = wall_distance)
    for vertex in object.rotated_vertices:
        if vertex[0] > wall_distance:
            # Position du sommet par rapport au centre de masse
            relative_position = vertex - object.position
            # Vitesse du sommet (translation + rotation)
            vertex_velocity = object.velocity + np.cross(object.angular_velocity, relative_position)
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
                object.velocity += (scalar_impulse * normal) / mass
                # Appliquer l'impulsion angulaire
                object.angular_velocity += np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
            
            # Replacer le sommet sur le mur en ajustant la position du centre de masse
            object.position[0] = min(object.position[0], wall_distance - relative_position[0])

    # Optionnel : limiter la rotation pour éviter les dérives numériques
    object.rotation = np.mod(object.rotation, 2 * np.pi)

def update_floor_and_ramp(
        object: Cube3D,
        x_ramp_min: float, x_ramp_max: float,
        z_ramp_min: float, z_ramp_max: float,
        ramp_angle_degrees: float):
    """
    Gère les collisions avec :
    • le sol (y = 0)
    • une rampe inclinée (angle `ramp_angle_degrees`) dans
        [x_ramp_min, x_ramp_max] x [z_ramp_min, z_ramp_max]

    Paramètres
    ----------
    friction_coefficient : μ, coefficient de frottement cinétique (≈ 0.2‑0.6)
    restitution_coefficient    : e, coefficient de restitution (0 = pas de rebond)
    """

    # --- 1. dynamique libre ------------------------------------------------
    object.velocity += GRAVITY * DT
    object.position += object.velocity * DT
    object.rotation += object.angular_velocity * DT
    object.rotated_vertices = object.get_vertices()

    # --- 2. constantes physiques ------------------------------------------
    mass = 1.0
    I = np.array([
        (1/12)*mass*(object.y_length**2 + object.z_length**2),
        (1/12)*mass*(object.x_length**2 + object.z_length**2),
        (1/12)*mass*(object.x_length**2 + object.y_length**2)
    ])

    theta_rad     = math.radians(ramp_angle_degrees)
    tan_theta     = math.tan(theta_rad)
    cos_theta     = math.cos(theta_rad)
    sin_theta     = math.sin(theta_rad)
    normal_ramp   = np.array([-sin_theta, cos_theta, 0.0])
    normal_ramp  /= np.linalg.norm(normal_ramp)
    normal_floor  = np.array([0.0, 1.0, 0.0])

    # Recalculer les sommets après mise à jour
    object.rotated_vertices = object.get_vertices()
    
    # Trier les sommets du plus proche du sol au plus loin
    sorted_vertices = sorted(object.rotated_vertices, key=lambda v: v[1])
    # is_close_to_ground si le plus proche <= 0.1
    is_close_to_ground = sorted_vertices[0][1] <= 0.1
    # is_on_ground si au moins 3 sommets sont <= 0.1
    is_on_ground = sorted_vertices[2][1] <= 0.1

    # Si le cube est au sol, appliquer une stabilisation plus agressive
    if is_close_to_ground:
        # Réduire drastiquement les vitesses horizontales et angulaires
        object.velocity[0] *= 0.9
        object.velocity[2] *= 0.9
        object.angular_velocity *= 0.9
    
    if is_on_ground:
        # Réduire drastiquement les vitesses horizontales et angulaires
        object.velocity[0] *= 0.2
        object.velocity[2] *= 0.2
        object.angular_velocity *= 0.1  # Réduction très forte quand sur le sol

        # Réduire drastiquement la vitesse verticale
        object.velocity[1] *= 0.2

    # --- 3. boucle sommets sur le plat -------------------------------------------------

    # Pour chaque sommet, vérifier la collision avec le sol
    for vertex in object.rotated_vertices:
        if vertex[1] < 0 and (vertex[0] < x_ramp_min or vertex[0] > x_ramp_max or vertex[2] < z_ramp_min or vertex[2] > z_ramp_max):
            # Position du sommet par rapport au centre de masse (c'est ce qui permet de casser la symétrie de la force d'impulsion)
            relative_position = vertex - object.position
            # Vitesse du sommet (translation + rotation)
            vertex_velocity = object.velocity + np.cross(object.angular_velocity, relative_position)
            # Si le sommet descend, on annule la composante verticale
            if vertex_velocity[1] < 0:
                # Impulsion nécessaire pour annuler la vitesse verticale
                relative_velocity = np.dot(vertex_velocity, normal_floor)
                # Calcul de l'impulsion scalaire
                r_cross_n = np.cross(relative_position, normal_floor)
                denom = (1/mass) + np.dot(normal_floor, np.cross(np.divide(r_cross_n, I, out=np.zeros_like(r_cross_n), where=I!=0), relative_position))
                if denom == 0:
                    continue
                scalar_impulse = -relative_velocity / denom
                # Appliquer l'impulsion au centre de masse
                object.velocity += (scalar_impulse * normal_floor) / mass
                # Appliquer l'impulsion angulaire
                object.angular_velocity += np.divide(np.cross(relative_position, scalar_impulse * normal_floor), I, out=np.zeros(3), where=I!=0)
            
            # Replacer le sommet sur le sol en ajustant la position du centre de masse
            object.position[1] = max(object.position[1], -relative_position[1])

    # --- 4. boucle sommets sur la rampe -------------------------------------------------
    for vertex in object.rotated_vertices:
        x, y, z = vertex
        if (x_ramp_min <= x <= x_ramp_max and z_ramp_min <= z <= z_ramp_max):
            ramp_y = (x - x_ramp_min) * tan_theta
            if y < ramp_y:
                relative_position = vertex - object.position
                vertex_velocity = object.velocity + np.cross(object.angular_velocity, relative_position)
                if np.dot(vertex_velocity, normal_ramp) < 0:
                    relative_velocity = np.dot(vertex_velocity, normal_ramp)
                    r_cross_n = np.cross(relative_position, normal_ramp)
                    denom = (1/mass) + np.dot(normal_ramp, np.cross(np.divide(r_cross_n, I, out=np.zeros_like(r_cross_n), where=I!=0), relative_position))
                    if denom == 0:
                        continue
                    scalar_impulse = -relative_velocity / denom
                    object.velocity += (scalar_impulse * normal_ramp) / mass
                    object.angular_velocity += np.divide(np.cross(relative_position, scalar_impulse * normal_ramp), I, out=np.zeros(3), where=I!=0)
                penetration = ramp_y - y                       # distance sous le plan
                if penetration > 0:
                    object.position += normal_ramp * (penetration / normal_ramp[1]) 

    # --- 5. borne les angles ----------------------------------------------
    object.rotation = np.mod(object.rotation, 2 * math.pi)


def update_on_stairs(object: Cube3D, stairs_coordinates_flat: dict[int, list[tuple[float, float]]], stairs_coordinates_vertical: dict[int, list[tuple[float, float, float]]], points_per_edge: int = 3):
        """
        Collision avancée entre le cube (les sommets et points intermédiaires pour une meilleure précision)
        et les marches d'escalier. Chaque marche est traitée comme un petit sol avec sa hauteur.
        Les contremarches verticales sont également prises en compte.

        stairs_coordinates_flat : dict[int, list[tuple[float, float]]]
        Chaque dict contient:
        - y: hauteur de la marche
        - step_points: liste de 4 points (x,z) définissant la marche (rectangle parallèle au sol)

        stairs_coordinates_vertical : dict[int, list[tuple[float, float, float]]]
        Chaque dict contient:
        - y: hauteur de la marche dont elle est la contremarche
        - step_points: liste de 4 points (x,y,z) définissant la contremarche (rectangle vertical)
        
        points_per_edge : int
        Nombre de points intermédiaires à ajouter sur chaque arête du cube pour améliorer la précision
        """
        # TODO:
        pass

def calculate_quadruped_inertia_tensor(quadruped: Quadruped):
    """
    Calcule le tenseur d'inertie du quadruped en tenant compte de la masse relative
    de chaque partie (body, upper legs, lower legs) et de leurs positions actuelles.
    
    Returns:
        np.array: Tenseur d'inertie [I_xx, I_yy, I_zz] au centre de masse
    """
    # Masse totale du quadruped
    total_mass = 1.0
    
    # Définir les masses relatives de chaque partie (basées sur le volume)
    # Body: 4.0 * 6.0 * 1.0 = 24.0
    # Upper legs: 4 * (1.0 * 1.0 * 1.0) = 4.0
    # Lower legs: 4 * (1.0 * 0.8 * 2.0) = 6.4
    # Total volume = 34.4
    
    body_volume = 4.0 * 6.0 * 1.0
    upper_leg_volume = 4 * (1.0 * 1.0 * 1.0)
    lower_leg_volume = 4 * (1.0 * 0.8 * 2.0)
    total_volume = body_volume + upper_leg_volume + lower_leg_volume
    
    body_mass = total_mass * (body_volume / total_volume)
    upper_leg_mass = total_mass * (upper_leg_volume / total_volume)
    lower_leg_mass = total_mass * (lower_leg_volume / total_volume)
    
    # Masse par partie individuelle
    single_upper_leg_mass = upper_leg_mass / 4
    single_lower_leg_mass = lower_leg_mass / 4
    
    # Initialiser le tenseur d'inertie total
    I_total = np.zeros(3)  # [I_xx, I_yy, I_zz]
    
    # 1. Tenseur d'inertie du body (cube)
    body_length, body_width, body_height = 4.0, 6.0, 1.0
    I_body_xx = (1/12) * body_mass * (body_height**2 + body_width**2)
    I_body_yy = (1/12) * body_mass * (body_length**2 + body_width**2)
    I_body_zz = (1/12) * body_mass * (body_length**2 + body_height**2)
    
    # Calculer le centre de masse du body dans le repère monde
    body_center = quadruped.position  # Le body est centré sur la position du quadruped
    
    # Ajouter le tenseur d'inertie du body (théorème de Steiner)
    I_total += np.array([I_body_xx, I_body_yy, I_body_zz])
    
    # 2. Tenseur d'inertie des upper legs
    upper_leg_length, upper_leg_width, upper_leg_height = 1.0, 1.0, 1.0
    I_upper_xx = (1/12) * single_upper_leg_mass * (upper_leg_height**2 + upper_leg_width**2)
    I_upper_yy = (1/12) * single_upper_leg_mass * (upper_leg_length**2 + upper_leg_width**2)
    I_upper_zz = (1/12) * single_upper_leg_mass * (upper_leg_length**2 + upper_leg_height**2)
    
    # Calculer les centres de masse des upper legs
    for i in range(4):
        # Utiliser les vertices des upper legs pour calculer leur centre
        upper_leg_start = 8 + i * 8  # Les upper legs commencent après le body (8 vertices)
        upper_leg_vertices = quadruped.rotated_vertices[upper_leg_start:upper_leg_start + 8]
        
        # Centre de masse de cette upper leg
        upper_leg_center = np.mean(upper_leg_vertices, axis=0)
        
        # Distance du centre de masse de cette leg au centre de masse total
        r = upper_leg_center - body_center
        r_squared = np.dot(r, r)
        
        # Théorème de Steiner: I = I_cm + m * d²
        I_total += np.array([I_upper_xx, I_upper_yy, I_upper_zz]) + single_upper_leg_mass * r_squared
    
    # 3. Tenseur d'inertie des lower legs
    lower_leg_length, lower_leg_width, lower_leg_height = 1.0, 0.8, 2.0
    I_lower_xx = (1/12) * single_lower_leg_mass * (lower_leg_height**2 + lower_leg_width**2)
    I_lower_yy = (1/12) * single_lower_leg_mass * (lower_leg_length**2 + lower_leg_width**2)
    I_lower_zz = (1/12) * single_lower_leg_mass * (lower_leg_length**2 + lower_leg_height**2)
    
    # Calculer les centres de masse des lower legs
    for i in range(4):
        # Utiliser les vertices des lower legs
        lower_leg_start = 40 + i * 8  # Les lower legs commencent après les upper legs (8 + 4*8 = 40)
        lower_leg_vertices = quadruped.rotated_vertices[lower_leg_start:lower_leg_start + 8]
        
        # Centre de masse de cette lower leg
        lower_leg_center = np.mean(lower_leg_vertices, axis=0)
        
        # Distance du centre de masse de cette leg au centre de masse total
        r = lower_leg_center - body_center
        r_squared = np.dot(r, r)
        
        # Théorème de Steiner: I = I_cm + m * d²
        I_total += np.array([I_lower_xx, I_lower_yy, I_lower_zz]) + single_lower_leg_mass * r_squared
    
    return I_total

def update_quadruped(quadruped: Quadruped):
    """
    Collision avancée entre le quadruped et le sol avec gestion améliorée des rebonds.
    """

    # Mettre à jour les sommets du cube
    quadruped.rotated_vertices = quadruped.get_vertices()

    # Constantes physiques
    mass = 1.0
    
    # Calculer le tenseur d'inertie précis basé sur la masse relative de chaque partie
    I = calculate_quadruped_inertia_tensor(quadruped)

    # Appliquer la gravité au centre de masse
    quadruped.velocity += GRAVITY * DT

    # Mise à jour de la position et de la rotation
    quadruped.position += quadruped.velocity * DT
    quadruped.rotation += quadruped.angular_velocity * DT
    
    # Recalculer les sommets après mise à jour
    quadruped.rotated_vertices = quadruped.get_vertices()
    
    # Trier les sommets du plus proche du sol au plus loin
    sorted_vertices = sorted(quadruped.rotated_vertices, key=lambda v: v[1])
    # is_close_to_ground si le plus proche <= 0.05
    is_close_to_ground = sorted_vertices[0][1] <= 0.05
    # is_on_ground si au moins 8 sommets sont <= 0.05 (réduit de 12 à 8)
    is_on_ground = sorted_vertices[7][1] <= 0.05

    # Limiter les vitesses maximales pour éviter la téléportation
    max_velocity = 10.0
    max_angular_velocity = 5.0
    
    quadruped.velocity = np.clip(quadruped.velocity, -max_velocity, max_velocity)
    quadruped.angular_velocity = np.clip(quadruped.angular_velocity, -max_angular_velocity, max_angular_velocity)

    # Si le quadruped est au sol, appliquer une stabilisation plus agressive
    if is_close_to_ground:
        # Réduire les vitesses horizontales et angulaires
        quadruped.velocity[0] *= 0.9
        quadruped.velocity[2] *= 0.9
        quadruped.angular_velocity *= 0.9
    
    if is_on_ground:
        # Réduire drastiquement les vitesses quand sur le sol
        quadruped.velocity[0] *= 0.3
        quadruped.velocity[1] *= 0.1  # Réduction très forte de la vitesse verticale
        quadruped.velocity[2] *= 0.3
        quadruped.angular_velocity *= 0.05  # Réduction très forte de la rotation

    # Collecter toutes les collisions avant de les appliquer
    total_impulse = np.zeros(3)
    total_angular_impulse = np.zeros(3)
    collision_count = 0
    
    # Pour chaque sommet, vérifier la collision avec le sol
    for vertex in quadruped.rotated_vertices:
        if vertex[1] < 0:
            # Position du sommet par rapport au centre de masse
            relative_position = vertex - quadruped.position
            # Vitesse du sommet (translation + rotation)
            vertex_velocity = quadruped.velocity + np.cross(quadruped.angular_velocity, relative_position)
            
            # Si le sommet descend, calculer l'impulsion nécessaire
            if vertex_velocity[1] < 0:
                # Impulsion nécessaire pour annuler la vitesse verticale
                normal = np.array([0, 1, 0])  # normale du sol
                relative_velocity = np.dot(vertex_velocity, normal)
                
                # Calcul de l'impulsion scalaire
                r_cross_n = np.cross(relative_position, normal)
                denom = (1/mass) + np.dot(normal, np.cross(np.divide(r_cross_n, I, out=np.zeros_like(r_cross_n), where=I!=0), relative_position))
                
                if denom != 0:
                    scalar_impulse = -relative_velocity / denom
                    
                    # Limiter l'impulsion pour éviter les rebonds excessifs
                    max_impulse = 5.0
                    scalar_impulse = np.clip(scalar_impulse, -max_impulse, max_impulse)
                    
                    # Accumuler les impulsions
                    total_impulse += (scalar_impulse * normal) / mass
                    total_angular_impulse += np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
                    collision_count += 1
            
            # Replacer le sommet sur le sol en ajustant la position du centre de masse
            quadruped.position[1] = max(quadruped.position[1], -relative_position[1])

    # Appliquer les impulsions moyennées si il y a eu des collisions
    if collision_count > 0:
        # Moyenner les impulsions pour éviter l'accumulation excessive
        average_impulse = total_impulse / collision_count
        average_angular_impulse = total_angular_impulse / collision_count
        
        # Limiter encore plus les impulsions moyennes
        max_average_impulse = 2.0
        average_impulse = np.clip(average_impulse, -max_average_impulse, max_average_impulse)
        average_angular_impulse = np.clip(average_angular_impulse, -max_average_impulse, max_average_impulse)
        
        quadruped.velocity += average_impulse
        quadruped.angular_velocity += average_angular_impulse

    # Optionnel : limiter la rotation pour éviter les dérives numériques
    quadruped.rotation = np.mod(quadruped.rotation, 2 * np.pi)