# update_functions.py
import numpy as np


from .config import DT, GRAVITY, RESTITUTION, FRICTION, CONTACT_THRESHOLD_BASE, CONTACT_THRESHOLD_MULTIPLIER
from .config import MAX_VELOCITY, MAX_ANGULAR_VELOCITY, MAX_IMPULSE, MAX_AVERAGE_IMPULSE, DEBUG_CONTACT, STATIC_FRICTION_CAP
from .quadruped import Quadruped
from .helpers import limit_vector


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
    mass = 5.0
    
    # Calculer le tenseur d'inertie à partir des vertices
    vertices = quadruped.rotated_vertices
    x_coords = [v[0] for v in vertices]
    y_coords = [v[1] for v in vertices]
    z_coords = [v[2] for v in vertices]
    
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords) * 0.8 # Le body du quadruped est situé en haut
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
    prev_vertices = quadruped.prev_vertices if quadruped.prev_vertices is not None else quadruped.rotated_vertices
    
    # Critères de contact dynamiques
    contact_threshold = max(CONTACT_THRESHOLD_BASE, abs(quadruped.velocity[1]) * DT * CONTACT_THRESHOLD_MULTIPLIER)

    # Limitation de vitesse douce
    quadruped.velocity = limit_vector(quadruped.velocity, MAX_VELOCITY)
    quadruped.angular_velocity = limit_vector(quadruped.angular_velocity, MAX_ANGULAR_VELOCITY)

    # Calculer la pénétration maximale sur tous les sommets
    penetrations = []

    # --- on sépare maintenant vertical (normal) et tangentiel ---
    collision_impulses_normal = []
    collision_angular_impulses_normal = []
    collision_impulses_tangent = []
    collision_angular_impulses_tangent = []
    
    for vertex in quadruped.rotated_vertices:
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
                    
                    # Limiter l'impulsion pour éviter les rebonds excessifs
                    if DEBUG_CONTACT:
                        print(f"[NORMAL] v_rel={relative_velocity:.3f}  "
                              f"jn_raw={scalar_impulse:.3f}")
                    scalar_impulse = np.clip(scalar_impulse, -MAX_IMPULSE, MAX_IMPULSE)
                    
                    # Stocker les impulsions pour application ultérieure
                    collision_impulses_normal.append((scalar_impulse * normal) / mass)
                    collision_angular_impulses_normal.append(
                        np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
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

    # --- Ajout : traction latérale basée sur t‑1 ---
    traction_imp, traction_ang = [], []
    for previous_vertex, current_vertex in zip(prev_vertices, quadruped.rotated_vertices):
        # le point est (et était) au sol ?
        previous_on_ground = previous_vertex[1] <= contact_threshold
        current_on_ground = current_vertex[1] <= contact_threshold
        if not (previous_on_ground and current_on_ground):
            continue
        # déplacement réel du sommet (inclut la cinématique des membres)
        current_vertex[1], previous_vertex[1] = 0, 0 # Normalisation de la Hauteur, on considère que les deux points restent au sol pendant la cinématique
        delta = current_vertex - previous_vertex
        delta[1] = 0.0  # composante tangentielle
        if np.linalg.norm(delta) < 1e-8:
            continue
        
        # vitesse “imposée” au sol → impulsion opposée sur le corps
        J_needed = -mass * delta / DT  # N·s
        J_cap = STATIC_FRICTION_CAP * DT  # adhérence max
        J = np.clip(J_needed, -J_cap, J_cap)
        # linéaire
        traction_imp.append(J / mass)
        # angulaire
        r = current_vertex - quadruped.position
        traction_ang.append(
            np.divide(np.cross(r, J), I, out=np.zeros(3), where=I!=0)
        )
    if traction_imp:
        quadruped.velocity += limit_vector(np.mean(traction_imp, axis=0), MAX_AVERAGE_IMPULSE)
        quadruped.angular_velocity += limit_vector(np.mean(traction_ang, axis=0), MAX_AVERAGE_IMPULSE)

    # --- Mémorise l’état pour la frame suivante ---
    quadruped.prev_vertices = quadruped.rotated_vertices.copy()