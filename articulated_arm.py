import pygame
import numpy as np
import math
from pygame.locals import *
from config import *
from camera import Camera3D
from cube import Cube3D
from ground import Ground
from joint import Joint
    

# --- Initialisation Pygame ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Moteur Physique 3D - Pygame")
clock = pygame.time.Clock()

# --- Objets du monde ---
camera = Camera3D()
ground = Ground(size=20)
forearm = Cube3D(
    position=np.array([1.0, 8.0, 1.0]),
    x_length=3.0,
    y_length=1.0,
    z_length=1.0
)
biceps = Cube3D(
    position=np.array([4.2, 8.0, 1.0]),
    x_length=2.0,
    y_length=1.0,
    z_length=1.0
)
joint = Joint(
    object_1=forearm, 
    object_2=biceps, 
    face_1=1, 
    face_2=3,
    initial_angle=0.0  # Joint ouvert plat au début
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
        forearm.reset()
        biceps.reset()
    
    # Contrôles du joint
    if keys[K_r]:  # R = Plier le joint (diminuer l'angle)
        current_angle = joint.angle
        joint.set_angle(current_angle - 0.05)  # Plier de 0.05 radians
    if keys[K_f]:  # F = Déplier le joint (augmenter l'angle)
        current_angle = joint.angle
        joint.set_angle(current_angle + 0.05)  # Déplier de 0.05 radians
    
    # --- Mise à jour physique ---
    joint.update()
    forearm.update_ground_only_complex()
    biceps.update_ground_only_complex()
    
    # --- Rendu ---
    screen.fill(BLACK)
    
    # Dessiner le monde 3D
    ground.draw(screen, camera)
    ground.draw_axes(screen, camera)
    forearm.draw(screen, camera)
    biceps.draw(screen, camera)
    joint.draw(screen, camera)
    
    # --- Interface utilisateur ---
    font = pygame.font.Font(None, 24)
    
    # Informations de position
    pos_text = f"Position: ({forearm.position[0]:.2f}, {forearm.position[1]:.2f}, {forearm.position[2]:.2f})"
    vel_text = f"Vitesse: ({forearm.velocity[0]:.2f}, {forearm.velocity[1]:.2f}, {forearm.velocity[2]:.2f})"
    cam_text = f"Caméra: ({camera.position[0]:.1f}, {camera.position[1]:.1f}, {camera.position[2]:.1f})"
    joint_text = f"Angle joint: {math.degrees(joint.angle):.1f}°"
    
    pos_surface = font.render(pos_text, True, WHITE)
    vel_surface = font.render(vel_text, True, WHITE)
    cam_surface = font.render(cam_text, True, WHITE)
    joint_surface = font.render(joint_text, True, WHITE)
    
    screen.blit(pos_surface, (10, 10))
    screen.blit(vel_surface, (10, 35))
    screen.blit(cam_surface, (10, 60))
    screen.blit(joint_surface, (10, 85))
    
    # Instructions
    instructions = [
        "Contrôles:",
        "ZQSD - Déplacer caméra",
        "AE - Monter/Descendre caméra", 
        "Flèches - Rotation caméra",
        "R/F - Plier/Déplier le joint",
        "Espace - Reset cube",
        "Échap - Quitter"
    ]
    
    for i, instruction in enumerate(instructions):
        inst_surface = font.render(instruction, True, GRAY)
        screen.blit(inst_surface, (10, WINDOW_HEIGHT - 120 + i * 20))
    
    # --- Mise à jour écran ---
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
