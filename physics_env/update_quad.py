# update_functions.py
import numpy as np


from .config import DT, GRAVITY, SLIP_THRESHOLD, CONTACT_THRESHOLD_BASE, CONTACT_THRESHOLD_MULTIPLIER
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
    quadruped._needs_update = True
    current_vertices = quadruped.get_vertices()
    prev_vertices = quadruped.prev_vertices if quadruped.prev_vertices is not None else current_vertices

    
    # --- paramètres corps -----------------------------
    mass = quadruped.mass          # ≈ 4.4 kg
    I    = quadruped.I_body.copy() # (3,)

    # Appliquer la gravité au centre de masse
    quadruped.velocity += GRAVITY * DT

    # Mise à jour de la position et de la rotation
    quadruped.position += quadruped.velocity * DT
    quadruped.rotation = (quadruped.rotation + quadruped.angular_velocity * DT) % (2 * np.pi)
    
    # Recalculer les sommets après mise à jour
    current_vertices = quadruped.get_vertices()
    
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
    
    for vertex in current_vertices:
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

                    scalar_impulse = np.clip(scalar_impulse, -MAX_IMPULSE, MAX_IMPULSE)
                    
                    # Stocker les impulsions pour application ultérieure
                    collision_impulses_normal.append((scalar_impulse * normal) / mass)
                    collision_angular_impulses_normal.append(
                        np.divide(np.cross(relative_position, scalar_impulse * normal), I, out=np.zeros(3), where=I!=0)
                    )
    
    # Appliquer la correction de position une seule fois
    if penetrations:
        mean_penetration = np.mean(penetrations)
        if DEBUG_CONTACT:
            print(f"[CORRECTION] Correction de position appliquée: +{mean_penetration * 0.2:.5f}")
        quadruped.position[1] += mean_penetration * 0.2

    # --- Moyenne et application des impulsions verticales ---
    if collision_impulses_normal:
        avg_imp_n = limit_vector(np.mean(collision_impulses_normal, axis=0), MAX_AVERAGE_IMPULSE)
        avg_ang_n = limit_vector(np.mean(collision_angular_impulses_normal, axis=0), MAX_AVERAGE_IMPULSE)
        if DEBUG_CONTACT:
            print(f"[IMPULSES N] Moyenne impulsion normale: {avg_imp_n}, angulaire: {avg_ang_n}")
        quadruped.velocity += avg_imp_n
        quadruped.angular_velocity += avg_ang_n

    # --- Moyenne et application des impulsions tangentielles ---
    if collision_impulses_tangent:
        avg_imp_t = limit_vector(np.mean(collision_impulses_tangent, axis=0), MAX_AVERAGE_IMPULSE)   # mets un plafond dédié si besoin
        avg_ang_t = limit_vector(np.mean(collision_angular_impulses_tangent, axis=0), MAX_AVERAGE_IMPULSE)
        if DEBUG_CONTACT:
            print(f"[IMPULSES T] Moyenne impulsion tangentielle: {avg_imp_t}, angulaire: {avg_ang_t}")
        quadruped.velocity += avg_imp_t
        quadruped.angular_velocity += avg_ang_t

    # --- Ajout : traction latérale basée sur t‑1 ---
    traction_imp, traction_ang = [], []
    for previous_vertex, current_vertex in zip(prev_vertices, current_vertices):
        # le point est (et était) au sol ?
        previous_on_ground = previous_vertex[1] <= contact_threshold
        current_on_ground = current_vertex[1] <= contact_threshold
        if not (previous_on_ground and current_on_ground):
            continue
        # déplacement réel du sommet (inclut la cinématique des membres)
        current_vertex[1], previous_vertex[1] = 0, 0 # Normalisation de la Hauteur, on considère que les deux points restent au sol pendant la cinématique
        delta = current_vertex - previous_vertex
        delta[1] = 0.0  # composante tangentielle
        # --- DEAD ZONE : on ignore les déplacements < SLIP_THRESHOLD*DT ---
        if np.linalg.norm(delta) < SLIP_THRESHOLD * DT:
            continue
        # vitesse "imposée" au sol → impulsion opposée sur le corps
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
        if DEBUG_CONTACT:
            print(f"[TRACTION] Moyenne traction linéaire: {np.mean(traction_imp, axis=0)}, angulaire: {np.mean(traction_ang, axis=0)}")
        quadruped.velocity += limit_vector(np.mean(traction_imp, axis=0), MAX_AVERAGE_IMPULSE)
        quadruped.angular_velocity += limit_vector(np.mean(traction_ang, axis=0), MAX_AVERAGE_IMPULSE)

    # Sauvegarder les vertices actuels pour la prochaine itération
    quadruped.prev_vertices = current_vertices.copy()

    if DEBUG_CONTACT:
        print(f"[VELOCITY] Velocity: {quadruped.velocity}, Angular Velocity: {quadruped.angular_velocity}")
        print(f"[POSITION] Position: {quadruped.position}, Rotation: {quadruped.rotation}")
        print("------------------------------------------------------------------------------------------------\n")