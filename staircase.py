import pygame
import numpy as np
import math
from pygame.locals import *
from config import *
from camera import Camera3D
from cube import Cube3D
from ground import Staircase        
    

# --- Initialisation Pygame ---
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Moteur Physique 3D - Pygame")
clock = pygame.time.Clock()

# --- Objets du monde ---
camera = Camera3D()
cube = Cube3D(
        position=np.array([1.2, 8.2, 1.2]),
        x_length=8.0,
        y_length=6.0,
        z_length=4.0,
        rotation=np.array([1.0, 1.0, 1.0])
    )
staircase = Staircase(size=20, num_steps=10, step_width=1.0, step_height=1.0, step_depth=1.0, start_x=0, start_z=0)

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
        cube.reset()
    
    # --- Mise à jour physique ---
    cube.update_on_stais(staircase.get_step_coordinates_flat())
    
    # --- Rendu ---
    screen.fill(BLACK)
    
    # Dessiner le monde 3D
    staircase.draw(screen, camera)
    staircase.draw_axes(screen, camera)
    staircase.draw_step_coordinates_vertical(screen, camera)  # Visualiser les coordonnées des contremarches
    staircase.draw_step_coordinates_flat(screen, camera)  # Visualiser les coordonnées des marches
    cube.draw(screen, camera)
    
    # --- Interface utilisateur ---
    font = pygame.font.Font(None, 24)
    
    # Informations de position
    pos_text = f"Position: ({cube.position[0]:.2f}, {cube.position[1]:.2f}, {cube.position[2]:.2f})"
    vel_text = f"Vitesse: ({cube.velocity[0]:.2f}, {cube.velocity[1]:.2f}, {cube.velocity[2]:.2f})"
    cam_text = f"Caméra: ({camera.position[0]:.1f}, {camera.position[1]:.1f}, {camera.position[2]:.1f})"
    
    pos_surface = font.render(pos_text, True, WHITE)
    vel_surface = font.render(vel_text, True, WHITE)
    cam_surface = font.render(cam_text, True, WHITE)
    
    screen.blit(pos_surface, (10, 10))
    screen.blit(vel_surface, (10, 35))
    screen.blit(cam_surface, (10, 60))
    
    # Instructions
    instructions = [
        "Contrôles:",
        "ZQSD - Déplacer caméra",
        "AE - Monter/Descendre caméra", 
        "Flèches - Rotation caméra",
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
