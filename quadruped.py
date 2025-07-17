# quadruped.py
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
body = Cube3D(
    position=np.array([0.0, 10.0, 0.0]),
    x_length=4.0,
    y_length=1.0,
    z_length=6.0
)

upper_legs = []
lower_legs = []
upper_leg_positions = [
    np.array([ 2.0, 8.5,  3.0]),  # Front right
    np.array([ 2.0, 8.5, -3.0]),  # Back right
    np.array([-2.0, 8.5,  3.0]),  # Front left
    np.array([-2.0, 8.5, -3.0]),  # Back left
]
lower_leg_positions = [
    pos + np.array([0.0, -2.0, 0.0]) for pos in upper_leg_positions
]

for i in range(4):
    upper_legs.append(Cube3D(
        position=upper_leg_positions[i].copy(),
        x_length=1.0,
        y_length=1.0,
        z_length=1.0,
        color = [0, 0, 255]
    ))
    lower_legs.append(Cube3D(
        position=lower_leg_positions[i].copy(),
        x_length=1.0,
        y_length=2.0,
        z_length=1.0,
        color = [0, 255, 0]
    ))

# Joint all upper legs to lower legs
joints = []
for i in range(4):
    joint = Joint(
        object_1=upper_legs[i],
        object_2=lower_legs[i],
        face_1=2,  # bottom of upper leg
        face_2=0,  # top of lower leg
        color = [255, i * 80, 0]
    )
    joints.append(joint)

# Joint all upper legs to body
corner_indices = [5, 4, 1, 0]
for i in range(4):
    joint = Joint(
        object_1=body,
        object_2=upper_legs[i],
        corner_1=corner_indices[i],
        face_2=0,
        color = [i * 80, 0, 255]
    )
    joints.append(joint)

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
        body.reset()
        for upper_leg in upper_legs:
            upper_leg.reset()
        for lower_leg in lower_legs:
            lower_leg.reset()
    
    # Contrôles du joint
    if keys[K_r]:  # R = Plier le joint (diminuer l'angle)
        current_angle = joints[0].angle
        joints[0].set_angle(current_angle - 0.05)  # Plier de 0.05 radians
    if keys[K_f]:  # F = Déplier le joint (augmenter l'angle)
        current_angle = joints[0].angle
        joints[0].set_angle(current_angle + 0.05)  # Déplier de 0.05 radians
    if keys[K_t]:  # T = Plier le joint (diminuer l'angle)
        current_angle = joints[0].angle
        for joint in joints:
            joint.set_angle(current_angle - 0.05)  # Plier de 0.05 radians
    if keys[K_g]:  # G = Déplier le joint (augmenter l'angle)
        current_angle = joints[0].angle
        for joint in joints:
            joint.set_angle(current_angle + 0.05)  # Déplier de 0.05 radians
    
    # --- Mise à jour physique ---
    for joint in joints:
        joint.update()
    
    for upper_leg in upper_legs:
        update_two_objects_with_joint(upper_leg, body, False)
    for i in range(4):
        update_two_objects_with_joint(lower_legs[i], upper_legs[i], False)
    
    for joint in joints:
        joint.update()
    
    
    # --- Rendu ---
    screen.fill(BLACK)
    
    # Dessiner le monde 3D
    ground.draw(screen, camera)
    ground.draw_axes(screen, camera)
    body.draw(screen, camera)
    for upper_leg in upper_legs:
        upper_leg.draw(screen, camera)
    for lower_leg in lower_legs:
        lower_leg.draw(screen, camera)
    for joint in joints:
        joint.draw(screen, camera)
    
    # --- Interface utilisateur ---
    font = pygame.font.Font(None, 24)
    
    # Informations de position
    pos_text = f"Position: ({body.position[0]:.2f}, {body.position[1]:.2f}, {body.position[2]:.2f})"
    vel_text = f"Vitesse: ({body.velocity[0]:.2f}, {body.velocity[1]:.2f}, {body.velocity[2]:.2f})"
    cam_text = f"Caméra: ({camera.position[0]:.1f}, {camera.position[1]:.1f}, {camera.position[2]:.1f})"
    joint_text = f"Angle joint: {math.degrees(joints[0].angle):.1f}°"
    
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
