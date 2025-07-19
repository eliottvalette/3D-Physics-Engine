# quadruped.py
import pygame
import numpy as np
import math
from pygame.locals import *
from config import *
from camera import Camera3D
from quadruped import Quadruped
from quadruped_points import get_quadruped_vertices, create_quadruped_vertices
from ground import Ground
from update_functions import *

# --- Initialisation Pygame ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Moteur Physique 3D - Pygame")
clock = pygame.time.Clock()

# --- Objets du monde ---
camera = Camera3D()
ground = Ground(size=20)

# Créer le quadruped avec les positions d'épaules calculées
quadruped_vertices_dict = create_quadruped_vertices()
quadruped = Quadruped(
    position=np.array([0.0, 5.5, 0.0]),
    vertices=get_quadruped_vertices(),
    vectrices_dict=quadruped_vertices_dict
)

# --- Contrôles caméra ---
camera_speed = 0.1
rotation_speed = 0.02

# --- Boucle principale ---
running = True
while running:
    # --- Gestion des événements ---
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
    
    # --- Contrôles caméra ---
    keys = pygame.key.get_pressed()
    
    # Mouvement caméra
    if keys[K_z]: # Z = Avancer
        camera.go_straight(camera_speed)
    if keys[K_s]: # S = Reculer
        camera.go_backward(camera_speed)
    if keys[K_q]: # Q = Gauche
        camera.go_left(camera_speed)
    if keys[K_d]: # D = Droite
        camera.go_right(camera_speed)
    if keys[K_a]: # A = Monter
        camera.position[1] += camera_speed
    if keys[K_e]: # E = Descendre
        camera.position[1] -= camera_speed
    
    # Rotation caméra
    if keys[K_LEFT]:
        camera.rotation[1] -= rotation_speed
    if keys[K_RIGHT]:
        camera.rotation[1] += rotation_speed
    if keys[K_UP]:
        camera.rotation[0] += rotation_speed
    if keys[K_DOWN]:
        camera.rotation[0] -= rotation_speed

    if keys[K_SPACE]:
        quadruped.reset()
    
    if keys[K_h]:
        print(quadruped.get_vertices())
    
    # --- Contrôles des articulations ---
    # Épaules (Front Right, Front Left, Back Right, Back Left)
    if keys[K_r]:  # Front Right shoulder
        quadruped.adjust_shoulder_angle(0, 0.05)
    if keys[K_f]:  # Front Right shoulder
        quadruped.adjust_shoulder_angle(0, -0.05)
    if keys[K_t]:  # Front Left shoulder
        quadruped.adjust_shoulder_angle(1, 0.05)
    if keys[K_g]:  # Front Left shoulder
        quadruped.adjust_shoulder_angle(1, -0.05)
    if keys[K_y]:  # Back Right shoulder
        quadruped.adjust_shoulder_angle(2, 0.05)
    if keys[K_h]:  # Back Right shoulder
        quadruped.adjust_shoulder_angle(2, -0.05)
    if keys[K_u]:  # Back Left shoulder
        quadruped.adjust_shoulder_angle(3, 0.05)
    if keys[K_j]:  # Back Left shoulder
        quadruped.adjust_shoulder_angle(3, -0.05)
    
    # Coudes (Front Right, Front Left, Back Right, Back Left)
    if keys[K_1]:  # Front Right elbow
        quadruped.adjust_elbow_angle(0, 0.05)
    if keys[K_2]:  # Front Right elbow
        quadruped.adjust_elbow_angle(0, -0.05)
    if keys[K_3]:  # Front Left elbow
        quadruped.adjust_elbow_angle(1, 0.05)
    if keys[K_4]:  # Front Left elbow
        quadruped.adjust_elbow_angle(1, -0.05)
    if keys[K_5]:  # Back Right elbow
        quadruped.adjust_elbow_angle(2, 0.05)
    if keys[K_6]:  # Back Right elbow
        quadruped.adjust_elbow_angle(2, -0.05)
    if keys[K_7]:  # Back Left elbow
        quadruped.adjust_elbow_angle(3, 0.05)
    if keys[K_8]:  # Back Left elbow
        quadruped.adjust_elbow_angle(3, -0.05)
    
    # Reset des articulations
    if keys[K_b]:
        quadruped.shoulder_angles = np.array([0.0, 0.0, 0.0, 0.0])
        quadruped.elbow_angles = np.array([0.0, 0.0, 0.0, 0.0])
        quadruped.rotated_vertices = quadruped.get_vertices()
    
    # --- Mise à jour physique ---
    update_quadruped(quadruped)
    
    # --- Rendu ---
    screen.fill(BLACK)
    
    # Dessiner le monde 3D
    ground.draw_premium(screen, camera)
    ground.draw_axes(screen, camera)
    quadruped.draw_premium(screen, camera)
    
    # --- Interface utilisateur ---
    font = pygame.font.Font(None, 24)
    
    # Informations de position
    pos_text = f"Position: ({quadruped.position[0]:.2f}, {quadruped.position[1]:.2f}, {quadruped.position[2]:.2f})"
    vel_text = f"Vitesse: ({quadruped.velocity[0]:.2f}, {quadruped.velocity[1]:.2f}, {quadruped.velocity[2]:.2f})"
    cam_text = f"Caméra: ({camera.position[0]:.1f}, {camera.position[1]:.1f}, {camera.position[2]:.1f})"
    rot_text = f"Rotation: ({quadruped.rotation[0]:.2f}, {quadruped.rotation[1]:.2f}, {quadruped.rotation[2]:.2f})"
    
    # Informations des articulations
    shoulder_text = f"Épaules: FR({quadruped.shoulder_angles[0]:.2f}) FL({quadruped.shoulder_angles[1]:.2f}) BR({quadruped.shoulder_angles[2]:.2f}) BL({quadruped.shoulder_angles[3]:.2f})"
    elbow_text = f"Coudes: FR({quadruped.elbow_angles[0]:.2f}) FL({quadruped.elbow_angles[1]:.2f}) BR({quadruped.elbow_angles[2]:.2f}) BL({quadruped.elbow_angles[3]:.2f})"
    
    pos_surface = font.render(pos_text, True, WHITE)
    vel_surface = font.render(vel_text, True, WHITE)
    cam_surface = font.render(cam_text, True, WHITE)
    rot_surface = font.render(rot_text, True, WHITE)
    shoulder_surface = font.render(shoulder_text, True, WHITE)
    elbow_surface = font.render(elbow_text, True, WHITE)
    
    screen.blit(pos_surface, (10, 10))
    screen.blit(vel_surface, (10, 35))
    screen.blit(cam_surface, (10, 60))
    screen.blit(rot_surface, (10, 85))
    screen.blit(shoulder_surface, (10, 110))
    screen.blit(elbow_surface, (10, 135))
    
    # Instructions
    instructions = [
        "Contrôles:",
        "ZQSD - Déplacer caméra",
        "AE - Monter/Descendre caméra", 
        "Flèches - Rotation caméra",
        "Espace - Reset quadruped",
        "R - Reset articulations",
        "1-4 - Épaules (FR,FL,BR,BL)",
        "5-8 - Coudes (FR,FL,BR,BL)",
        "H - Afficher les sommets",
        "Échap - Quitter"
    ]
    
    for i, instruction in enumerate(instructions):
        inst_surface = font.render(instruction, True, GRAY)
        screen.blit(inst_surface, (10, WINDOW_HEIGHT - 140 + i * 20))
    
    # --- Mise à jour écran ---
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
