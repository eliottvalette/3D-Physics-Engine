# quadruped_env.py
import time
import pygame
import numpy as np
from pygame.locals import *

from physics_env.config import *
from physics_env.camera import Camera3D
from physics_env.quadruped import Quadruped
from physics_env.quadruped_points import get_quadruped_vertices, create_quadruped_vertices
from physics_env.ground import Ground
from physics_env.update_functions import update_quadruped
from physics_env.config import *

class QuadrupedEnv:
    """
    Main game class for simulating and controlling a quadruped robot in a 3D environment using Pygame.
    Handles camera movement, joint controls, rendering, and UI.
    """
    # Key mappings for joint controls
    SHOULDER_KEYS = [
        (K_r, 0, 1), (K_f, 0, -1),  # Front Right
        (K_t, 2, 1), (K_g, 2, -1),  # Front Left
        (K_y, 1, 1), (K_h, 1, -1),  # Back Right
        (K_u, 3, 1), (K_j, 3, -1),  # Back Left
    ]
    ELBOW_KEYS = [
        (K_1, 0, 1), (K_5, 0, -1),  # Front Right
        (K_2, 2, 1), (K_6, 2, -1),  # Front Left
        (K_3, 1, 1), (K_7, 1, -1),  # Back Right
        (K_4, 3, 1), (K_8, 3, -1),  # Back Left
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
    CIRCLE_RADII = [0.1, 0.3, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def __init__(self, rendering=True):
        """Initialize the game, Pygame, and world objects."""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Moteur Physique 3D - Pygame")
        self.clock = pygame.time.Clock()
        self.rendering = rendering

        # World objects
        self.camera = Camera3D()
        self.ground = Ground(size=20)
        self.quadruped_vertices_dict = create_quadruped_vertices()
        self.quadruped = Quadruped(
            position=np.array([0.0, 5.5, 0.0]),
            vertices=get_quadruped_vertices(),
            vertices_dict=self.quadruped_vertices_dict
        )
        self.camera_speed = 0.1
        self.rotation_speed = 0.02
        self.font = pygame.font.Font(None, 24)

        self.circle_radii   = self.CIRCLE_RADII
        self.circles_passed = set()         # stocke les rayons déjà comptés
        
        # Attributs pour la reward
        self.prev_potential = None        # pour ∆Φ
        self.potential_coef  = 3.0
        self.rot_penalty_coef = 0.5
        self.consecutive_steps_below_critical_height = 0
        
    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_events()
            keys = pygame.key.get_pressed()
            shoulder_actions, elbow_actions = self.handle_joint_controls(keys)
            camera_actions = self.handle_camera_controls(keys)
            reset_actions = [0, 0]
            if keys[K_SPACE]:
                reset_actions[0] = 1
            if keys[K_b]:
                reset_actions[1] = 1

            if keys[K_p]:
                print(f"state: {self.quadruped.get_state()}, len: {len(self.quadruped.get_state())}")
                time.sleep(0.1)
            
            _, reward, done, step_time = self.step(shoulder_actions, elbow_actions, camera_actions, reset_actions)

            update_quadruped(self.quadruped)
            if self.rendering:
                self.render(reward, done, step_time)
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
        camera_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Movement
        if keys[K_z]:
            camera_actions[0] = 1
        if keys[K_s]:
            camera_actions[1] = 1
        if keys[K_q]:
            camera_actions[2] = 1
        if keys[K_d]:
            camera_actions[3] = 1
        if keys[K_e]:
            camera_actions[4] = 1
        if keys[K_a]:
            camera_actions[5] = 1
        # Rotation
        if keys[K_LEFT]:
            camera_actions[6] = 1
        if keys[K_RIGHT]:
            camera_actions[7] = 1
        if keys[K_DOWN]:
            camera_actions[8] = 1
        if keys[K_UP]:
            camera_actions[9] = 1

        return camera_actions

    def handle_joint_controls(self, keys):
        """Handle joint (shoulder and elbow) controls based on key input."""
        shoulder_actions = [0, 0, 0, 0]
        elbow_actions = [0, 0, 0, 0]
        for key, idx, sign in self.SHOULDER_KEYS:
            if keys[key]:
                shoulder_actions[idx] = sign
        for key, idx, sign in self.ELBOW_KEYS:
            if keys[key]:
                elbow_actions[idx] = sign
        return shoulder_actions, elbow_actions
    
    def step(self, shoulder_actions, elbow_actions, camera_actions = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], reset_actions = [0, 0]):
        """Step the quadruped in the environment.
        
        actions_array is a vector composed only of 0 and 1.
        --- Selected by the agent ---
        The first 4 elements are the actions for the shoulders. (FR, FL, BR, BL)
        The following 4 elements are the actions for the elbows. (FR, FL, BR, BL)
        --- Only in manual mode ---
        The following 10 are for the camera. (z, s, q, d, e, a, left, right, up, down)
        The following 2 are for the reset (no-rd and rd). (space and b)
        The following 1 for the vertices display. (p)
        
        The action is the following:
        - 0: no action
        - 1: action
        """

        start_step_time = time.time()

        # Update all Shoulder and Elbow angles
        for idx, action in enumerate(shoulder_actions):
            self.quadruped.adjust_shoulder_angle(idx, SHOULDER_DELTA * action)
        for idx, action in enumerate(elbow_actions):
            self.quadruped.adjust_elbow_angle(idx, ELBOW_DELTA * action)

        # === un seul recalcul pour toutes les articulations ===
        _ = self.quadruped.get_vertices()

        # Update camera
        for idx, action in enumerate(camera_actions):
            if action:
                self.camera.update_camera(idx, self.camera_speed, self.rotation_speed)
        
        # Reset quadruped
        if reset_actions[0]:
            self.quadruped.reset_random()
            self.circles_passed.clear()
            self.prev_potential = None
        if reset_actions[1]:
            self.quadruped.reset()
            self.circles_passed.clear()
            self.prev_potential = None

        # Update quadruped
        update_quadruped(self.quadruped)

        next_state = self.get_state()
        # ---- REWARD ----------------------------------------------------- 
        # distance horizontale à l'origine
        radius = np.hypot(self.quadruped.position[0],
                          self.quadruped.position[2])

        # ----------  a)  Pénalité d'inclinaison du corps (pénalité brute) --------------
        pitch, _, roll = self.quadruped.rotation
        pitch = ((pitch + np.pi) % (2 * np.pi)) - np.pi  # Normaliser les angles entre -π et π
        roll = ((roll + np.pi) % (2 * np.pi)) - np.pi
        tilt_penalty = -self.rot_penalty_coef * (abs(pitch) + abs(roll))
        
        # ----------  b)  Shaping "potentiel" vers le prochain cercle --------------
        # distance horizontale actuelle
        # prochain cercle non encore franchi
        remaining = [r for r in self.circle_radii if r not in self.circles_passed]
        next_target = min(remaining) if remaining else self.circle_radii[-1]
        rho_t = abs(next_target - radius)          # distance à l'objectif courant
        phi_t = -rho_t                             # potentiel
        if self.prev_potential is None:
            delta_phi = 0.0                        # premier pas
        else:
            delta_phi = GAMMA * phi_t - self.prev_potential
        self.prev_potential = phi_t                # mise à jour pour le pas suivant
        
        # ----------  c)  Récompense principale --------------
        sparse_reward = 0.0
        for r in self.circle_radii:
            if radius >= r and r not in self.circles_passed:
                sparse_reward += 10.0
                self.circles_passed.add(r)
        
        below_critical_height = self.quadruped.position[1] < 4.0
        if below_critical_height:
            sparse_reward = -0.5

        if below_critical_height:
            self.consecutive_steps_below_critical_height += 1
        else:
            self.consecutive_steps_below_critical_height = 0

        if self.consecutive_steps_below_critical_height > 50:
            done = True
        else:
            done = False

        # ----------  d)  Somme finale -------------------------
        reward = sparse_reward + tilt_penalty + self.potential_coef * delta_phi
        # -----------------------------------------------------
        end_step_time = time.time()
        step_time = end_step_time - start_step_time

        return next_state, reward, done, step_time


    def render(self, reward, done = False, step_time = 0.0):
        """Render the 3D world and UI."""
        self.screen.fill(BLACK)
        self.ground.draw(self.screen, self.camera)
        self.ground.draw_axes(self.screen, self.camera)
        self.draw_checkpoint_circles()
        self.quadruped.draw(self.screen, self.camera)
        self.render_ui(reward, done, step_time)
        pygame.display.flip()
    
    def get_state(self):
        """Get the current state of the quadruped."""
        state = self.quadruped.get_state()
        return state

    def render_ui(self, reward, done = False, step_time = 0.0):
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
        reward_text = f"Récompense: {reward:.2f}"
        done_text = f"Terminé ? {done}"
        score_text = f"Score cercles: {len(self.circles_passed)}"
        step_time_text = f"Step time: {(step_time*1000):.6f}ms"
        surfaces = [
            self.font.render(pos_text, True, WHITE),
            self.font.render(vel_text, True, WHITE),
            self.font.render(cam_text, True, WHITE),
            self.font.render(rot_text, True, WHITE),
            self.font.render(shoulder_text, True, WHITE),
            self.font.render(elbow_text, True, WHITE),
            self.font.render(reward_text, True, WHITE),
            self.font.render(done_text, True, WHITE),
            self.font.render(score_text, True, WHITE),
            self.font.render(step_time_text, True, WHITE),
        ]
        for i, surf in enumerate(surfaces):
            self.screen.blit(surf, (10, 10 + i * 25))
        # Instructions
        for i, instruction in enumerate(self.INSTRUCTIONS):
            inst_surface = self.font.render(instruction, True, GRAY)
            self.screen.blit(inst_surface, (10, WINDOW_HEIGHT - 140 + i * 20))

    def draw_checkpoint_circles(self):
        """Dessine les cercles de récompense au sol."""
        segments = 36
        for r in self.circle_radii:
            pts = []
            for theta in np.linspace(0, 2*np.pi, segments, endpoint=False):
                world_pt = np.array([r*np.cos(theta), 0.0, r*np.sin(theta)])
                proj = self.camera.project_3d_to_2d(world_pt)
                if proj:                 # point visible
                    pts.append(proj[:2])
            if len(pts) > 1:
                color = (0, 255, 0) if r in self.circles_passed else (100, 100, 100)
                pygame.draw.lines(self.screen, color, True, pts, 1)

if __name__ == "__main__":
    game = QuadrupedEnv()
    game.run()
