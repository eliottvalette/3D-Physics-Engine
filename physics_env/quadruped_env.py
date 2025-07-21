# quadruped_env.py
import pygame
import numpy as np
from pygame.locals import *
from config import *
from camera import Camera3D
from quadruped import Quadruped
from quadruped_points import get_quadruped_vertices, create_quadruped_vertices
from ground import Ground
from update_functions import update_quadruped

class QuadrupedEnv:
    """
    Main game class for simulating and controlling a quadruped robot in a 3D environment using Pygame.
    Handles camera movement, joint controls, rendering, and UI.
    """
    # Key mappings for joint controls
    SHOULDER_KEYS = [
        (K_r, 0, 0.05), (K_f, 0, -0.05),  # Front Right
        (K_t, 1, 0.05), (K_g, 1, -0.05),  # Front Left
        (K_y, 2, 0.05), (K_h, 2, -0.05),  # Back Right
        (K_u, 3, 0.05), (K_j, 3, -0.05),  # Back Left
    ]
    ELBOW_KEYS = [
        (K_1, 0, 0.05), (K_2, 0, -0.05),  # Front Right
        (K_3, 1, 0.05), (K_4, 1, -0.05),  # Front Left
        (K_5, 2, 0.05), (K_6, 2, -0.05),  # Back Right
        (K_7, 3, 0.05), (K_8, 3, -0.05),  # Back Left
    ]
    INSTRUCTIONS = [
        "Contrôles:",
        "ZQSD - Déplacer caméra",
        "AE - Monter/Descendre caméra",
        "Flèches - Rotation caméra",
        "Espace - Reset quadruped",
        "B - Reset articulations",
        "R/F/T/G/Y/H/U/J - Épaules (FR,FL,BR,BL)",
        "1-8 - Coudes (FR,FL,BR,BL)",
        "P - Afficher les sommets",
        "Échap - Quitter"
    ]

    def __init__(self):
        """Initialize the game, Pygame, and world objects."""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Moteur Physique 3D - Pygame")
        self.clock = pygame.time.Clock()

        # World objects
        self.camera = Camera3D()
        self.ground = Ground(size=20)
        self.quadruped_vertices_dict = create_quadruped_vertices()
        self.quadruped = Quadruped(
            position=np.array([0.0, 5.5, 0.0]),
            vertices=get_quadruped_vertices(),
            vectrices_dict=self.quadruped_vertices_dict
        )
        self.camera_speed = 0.1
        self.rotation_speed = 0.02
        self.font = pygame.font.Font(None, 24)

    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_events()
            keys = pygame.key.get_pressed()
            self.handle_camera_controls(keys)
            self.handle_joint_controls(keys)
            if keys[K_SPACE]:
                self.quadruped.reset_random()
            if keys[K_p]:
                print(self.quadruped.get_vertices())
            if keys[K_b]:
                self.quadruped.reset()
            update_quadruped(self.quadruped)
            self.render()
            self.clock.tick(FPS)
        pygame.quit()

    def handle_events(self):
        """Handle Pygame events. Returns False if the game should exit."""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
        return True

    def handle_camera_controls(self, keys):
        """Handle camera movement and rotation based on key input."""
        # Movement
        if keys[K_z]:
            self.camera.go_straight(self.camera_speed)
        if keys[K_s]:
            self.camera.go_backward(self.camera_speed)
        if keys[K_q]:
            self.camera.go_left(self.camera_speed)
        if keys[K_d]:
            self.camera.go_right(self.camera_speed)
        if keys[K_e]:
            self.camera.position[1] += self.camera_speed
        if keys[K_a]:
            self.camera.position[1] -= self.camera_speed
        # Rotation
        if keys[K_LEFT]:
            self.camera.rotation[1] -= self.rotation_speed
        if keys[K_RIGHT]:
            self.camera.rotation[1] += self.rotation_speed
        if keys[K_UP]:
            self.camera.rotation[0] += self.rotation_speed
        if keys[K_DOWN]:
            self.camera.rotation[0] -= self.rotation_speed

    def handle_joint_controls(self, keys):
        """Handle joint (shoulder and elbow) controls based on key input."""
        for key, idx, delta in self.SHOULDER_KEYS:
            if keys[key]:
                self.quadruped.adjust_shoulder_angle(idx, delta)
        for key, idx, delta in self.ELBOW_KEYS:
            if keys[key]:
                self.quadruped.adjust_elbow_angle(idx, delta)

    def render(self):
        """Render the 3D world and UI."""
        self.screen.fill(BLACK)
        self.ground.draw_premium(self.screen, self.camera)
        self.ground.draw_axes(self.screen, self.camera)
        self.quadruped.draw_premium(self.screen, self.camera)
        self.render_ui()
        pygame.display.flip()
    
    def get_state(self):
        """Get the current state of the quadruped."""
        return self.quadruped.get_state()

    def render_ui(self):
        """Render the UI overlays (info and instructions)."""
        # Info texts
        pos_text = f"Position: ({self.quadruped.position[0]:.2f}, {self.quadruped.position[1]:.2f}, {self.quadruped.position[2]:.2f})"
        vel_text = f"Vitesse: ({self.quadruped.velocity[0]:.2f}, {self.quadruped.velocity[1]:.2f}, {self.quadruped.velocity[2]:.2f})"
        cam_text = f"Caméra: ({self.camera.position[0]:.1f}, {self.camera.position[1]:.1f}, {self.camera.position[2]:.1f})"
        rot_text = f"Rotation: ({self.quadruped.rotation[0]:.2f}, {self.quadruped.rotation[1]:.2f}, {self.quadruped.rotation[2]:.2f})"
        shoulder_text = (
            f"Épaules: FR({self.quadruped.shoulder_angles[0]:.2f}) "
            f"FL({self.quadruped.shoulder_angles[1]:.2f}) "
            f"BR({self.quadruped.shoulder_angles[2]:.2f}) "
            f"BL({self.quadruped.shoulder_angles[3]:.2f})"
        )
        elbow_text = (
            f"Coudes: FR({self.quadruped.elbow_angles[0]:.2f}) "
            f"FL({self.quadruped.elbow_angles[1]:.2f}) "
            f"BR({self.quadruped.elbow_angles[2]:.2f}) "
            f"BL({self.quadruped.elbow_angles[3]:.2f})"
        )
        surfaces = [
            self.font.render(pos_text, True, WHITE),
            self.font.render(vel_text, True, WHITE),
            self.font.render(cam_text, True, WHITE),
            self.font.render(rot_text, True, WHITE),
            self.font.render(shoulder_text, True, WHITE),
            self.font.render(elbow_text, True, WHITE),
        ]
        for i, surf in enumerate(surfaces):
            self.screen.blit(surf, (10, 10 + i * 25))
        # Instructions
        for i, instruction in enumerate(self.INSTRUCTIONS):
            inst_surface = self.font.render(instruction, True, GRAY)
            self.screen.blit(inst_surface, (10, WINDOW_HEIGHT - 140 + i * 20))

if __name__ == "__main__":
    game = QuadrupedEnv()
    game.run()
