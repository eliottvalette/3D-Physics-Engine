# double_articulated_arm.py
import pygame
import numpy as np
import math
from pygame.locals import *
from config import *
from camera import Camera3D
from cube import Cube3D
from ground import Ground
from joint import Joint
from update_functions import *

# --- Initialisation Pygame ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Moteur Physique 3D - Pygame")
clock = pygame.time.Clock()

# --- Objets du monde ---
camera = Camera3D()
ground = Ground(size=20)
first = Cube3D(
    position=np.array([1.0, 8.0, 1.0]),
    x_length=3.0,
    y_length=1.0,
    z_length=1.0,
    color=(255, 0, 0)
)
second = Cube3D(
    position=np.array([4.2, 8.0, 1.0]),
    x_length=2.0,
    y_length=1.0,
    z_length=1.0,
    color=(0, 255, 0)
)
third = Cube3D(
    position=np.array([7.4, 8.0, 1.0]),
    x_length=2.0,
    y_length=1.0,
    z_length=1.0,
    color=(0, 0, 255)
)
joint_1 = Joint(
    object_1=first, 
    object_2=second, 
    face_1=1, 
    face_2=3,
    initial_angle=0.0  # Joint ouvert plat au début
)
joint_2 = Joint(
    object_1=second, 
    object_2=third, 
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
        first.reset()
        second.reset()
        third.reset()
    
    # Contrôles du joint
    if keys[K_r]:  # R = Plier le joint (diminuer l'angle)
        current_angle = joint_1.angle
        joint_1.set_angle(current_angle - 0.05)  # Plier de 0.05 radians
    if keys[K_f]:  # F = Déplier le joint (augmenter l'angle)
        current_angle = joint_1.angle
        joint_1.set_angle(current_angle + 0.05)  # Déplier de 0.05 radians
    if keys[K_t]:  # T = Plier le joint (diminuer l'angle)
        current_angle = joint_2.angle
        joint_2.set_angle(current_angle - 0.05)  # Plier de 0.05 radians
    if keys[K_g]:  # G = Déplier le joint (augmenter l'angle)
        current_angle = joint_2.angle
        joint_2.set_angle(current_angle + 0.05)  # Déplier de 0.05 radians
    
    # --- Mise à jour physique ---
    joint_1.update()
    joint_2.update()
    update_two_objects_with_joint(first, second, False)
    update_two_objects_with_joint(second, third, True)
    
    # --- Rendu ---
    screen.fill(BLACK)
    
    # Dessiner le monde 3D
    ground.draw(screen, camera)
    ground.draw_axes(screen, camera)
    first.draw(screen, camera)
    second.draw(screen, camera)
    third.draw(screen, camera)
    joint_1.draw(screen, camera)
    joint_2.draw(screen, camera)
    
    # --- Interface utilisateur ---
    font = pygame.font.Font(None, 24)
    
    # Informations de position
    pos_text = f"Position: ({first.position[0]:.2f}, {first.position[1]:.2f}, {first.position[2]:.2f})"
    vel_text = f"Vitesse: ({first.velocity[0]:.2f}, {first.velocity[1]:.2f}, {first.velocity[2]:.2f})"
    cam_text = f"Caméra: ({camera.position[0]:.1f}, {camera.position[1]:.1f}, {camera.position[2]:.1f})"
    joint_text = f"Angle joint: {math.degrees(joint_1.angle):.1f}°"
    
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
        "T/G - Plier/Déplier le joint",
        "Espace - Reset cube",
        "Échap - Quitter"
    ]
    
    for i, instruction in enumerate(instructions):
        inst_surface = font.render(instruction, True, GRAY)
        screen.blit(inst_surface, (10, WINDOW_HEIGHT - 140 + i * 20))
    
    # --- Mise à jour écran ---
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
