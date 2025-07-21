# update_functions.py
import numpy as np
import math
import random

from config import DT, GRAVITY, RESTITUTION, FRICTION, CONTACT_THRESHOLD_BASE, CONTACT_THRESHOLD_MULTIPLIER
from config import MAX_VELOCITY, MAX_ANGULAR_VELOCITY, MAX_IMPULSE, MAX_AVERAGE_IMPULSE, DEBUG_CONTACT
from cube import Cube3D
from joint import Joint
from quadruped import Quadruped

def limit_vector(vec, max_val):
    """
    Limite un vecteur de manière douce en préservant sa direction
    """
    norm = np.linalg.norm(vec)
    if norm <= max_val:
        return vec
    else:
        return vec * (max_val / norm)

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

def update_quadruped(quadruped: Quadruped):
    """
    Collision avancée entre le quadruped et le sol avec gestion améliorée des rebonds.
    Corrections physiques appliquées :
    - Correction de pénétration conservatrice
    - Critères de contact dynamiques
    - Amortissement réaliste avec restitution et friction
    - Limitation de vitesse douce
    - Rotation stable
    """

    # Mettre à jour les sommets du cube
    quadruped.rotated_vertices = quadruped.get_vertices()

    # Constantes physiques
    mass = 1.0
    
    # Calculer le tenseur d'inertie à partir des vertices
    vertices = quadruped.rotated_vertices
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    z_range = max(z_coords) - min(z_coords)
    
    # Tenseur d'inertie approximatif basé sur les dimensions calculées
    I_xx = (1/12) * mass * (y_range**2 + z_range**2)
    I_yy = (1/12) * mass * (x_range**2 + z_range**2)
    I_zz = (1/12) * mass * (x_range**2 + y_range**2)
    
    I = np.array([I_xx, I_yy, I_zz])

    # Appliquer la gravité au centre de masse
    quadruped.velocity += GRAVITY * DT

    # Mise à jour de la position et de la rotation
    quadruped.position += quadruped.velocity * DT
    quadruped.rotation = (quadruped.rotation + quadruped.angular_velocity * DT) % (2 * np.pi)
    
    # Recalculer les sommets après mise à jour
    quadruped.rotated_vertices = quadruped.get_vertices()
    
    # Critères de contact dynamiques
    contact_threshold = max(CONTACT_THRESHOLD_BASE, abs(quadruped.velocity[1]) * DT * CONTACT_THRESHOLD_MULTIPLIER)
    
    # Trier les sommets du plus proche du sol au plus loin
    sorted_vertices = sorted(quadruped.rotated_vertices, key=lambda v: v[1])
    is_close_to_ground = sorted_vertices[0][1] <= contact_threshold
    is_on_ground = sorted_vertices[7][1] <= contact_threshold

    # Limitation de vitesse douce
    quadruped.velocity = limit_vector(quadruped.velocity, MAX_VELOCITY)
    quadruped.angular_velocity = limit_vector(quadruped.angular_velocity, MAX_ANGULAR_VELOCITY)

    # Amortissement réaliste avec restitution et friction
    if is_close_to_ground:
        # Réduire légèrement les vitesses horizontales et angulaires
        quadruped.velocity[0] *= (1 - FRICTION * 0.1)
        quadruped.velocity[2] *= (1 - FRICTION * 0.1)
        quadruped.angular_velocity *= (1 - FRICTION * 0.1)
    
    if is_on_ground:
        # Appliquer la restitution et la friction quand sur le sol
        quadruped.velocity[1] *= -RESTITUTION  # Rebond avec coefficient de restitution
        quadruped.velocity[0] *= (1 - FRICTION)  # Friction horizontale
        quadruped.velocity[2] *= (1 - FRICTION)  # Friction horizontale
        quadruped.angular_velocity *= (1 - FRICTION)  # Friction angulaire

    # Calculer la pénétration maximale sur tous les sommets
    penetrations = []

    # --- on sépare maintenant vertical (normal) et tangentiel ---
    collision_impulses_normal = []
    collision_angular_impulses_normal = []
    collision_impulses_tangent = []
    collision_angular_impulses_tangent = []
    
    for vertex in quadruped.rotated_vertices:
        jn_scalar = 0.0  # impulsion normale appliquée sur CE sommet (servira à la friction)
        if vertex[1] < 0:
            # Position du sommet par rapport au centre de masse
            relative_position = vertex - quadruped.position
            # Vitesse du sommet (translation + rotation)
            vertex_velocity = quadruped.velocity + np.cross(quadruped.angular_velocity, relative_position)
            
            # Enregistrer la pénétration
            penetrations.append(-vertex[1])
            
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
                    jn_scalar = scalar_impulse  # pour la friction
                    
                    # Limiter l'impulsion pour éviter les rebonds excessifs
                    scalar_impulse = np.clip(scalar_impulse, -MAX_IMPULSE, MAX_IMPULSE)
                    
                    # Stocker les impulsions pour application ultérieure
                    collision_impulses_normal.append((scalar_impulse * normal) / mass)
                    collision_angular_impulses_normal.append(
                        np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
                    )
        
         # ---- 3.b Impulsion tangentielle ------------------------
        if vertex[1] < 0.1:
            normal = np.array([0.0, 1.0, 0.0])
            relative_position = vertex - quadruped.position
            vertex_velocity = quadruped.velocity + np.cross(quadruped.angular_velocity, relative_position)

            v_normal = np.dot(vertex_velocity, normal) * normal
            v_tangent = vertex_velocity - v_normal
            v_tan_norm = np.linalg.norm(v_tangent)

            if v_tan_norm > 1e-6:
                # Direction unitaire tangentielle
                t_dir = v_tangent / v_tan_norm

                # Masse effective dans la direction tangentielle
                r_cross_t = np.cross(relative_position, t_dir)
                denom_t = (1/mass) + np.dot(
                    t_dir,
                    np.cross(
                        np.divide(r_cross_t, I, out=np.zeros_like(r_cross_t), where=I!=0),
                        relative_position
                    )
                )

                if denom_t != 0:
                    # Impulsion maximale autorisée par Coulomb basée sur l'impulsion normale réelle
                    mu = FRICTION
                    max_j_t = mu * abs(jn_scalar) if jn_scalar != 0 else mu * mass * np.linalg.norm(GRAVITY) * DT

                    # Impulsion nécessaire pour annuler la vitesse tangentielle
                    j_t_magnitude = min(v_tan_norm / denom_t, max_j_t)
                    j_t = -t_dir * j_t_magnitude

                    collision_impulses_tangent.append(j_t / mass)
                    collision_angular_impulses_tangent.append(
                        np.divide(np.cross(relative_position, j_t), I, out=np.zeros(3), where=I!=0)
                    )

    # Appliquer la correction de position une seule fois
    if penetrations:
        max_penetration = max(penetrations)
        quadruped.position[1] += max_penetration

    # --- Moyenne et application des impulsions verticales ---
    if collision_impulses_normal:
        avg_imp_n = limit_vector(np.mean(collision_impulses_normal, axis=0), MAX_AVERAGE_IMPULSE)
        avg_ang_n = limit_vector(np.mean(collision_angular_impulses_normal, axis=0), MAX_AVERAGE_IMPULSE)
        quadruped.velocity += avg_imp_n
        quadruped.angular_velocity += avg_ang_n

    # --- Moyenne et application des impulsions tangentielles ---
    if collision_impulses_tangent:
        avg_imp_t = limit_vector(np.mean(collision_impulses_tangent, axis=0), MAX_AVERAGE_IMPULSE)   # mets un plafond dédié si besoin
        avg_ang_t = limit_vector(np.mean(collision_angular_impulses_tangent, axis=0), MAX_AVERAGE_IMPULSE)
        quadruped.velocity += avg_imp_t
        quadruped.angular_velocity += avg_ang_t