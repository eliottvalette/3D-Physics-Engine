# showcase_video.py
from manim import *
import numpy as np
import shutil
import os


class Showcase(ThreeDScene):
    """Vidéo pédagogique : 6 sections, chacune ↦ 1 visuel + 1 équation."""

    # -----------------------------------------------------------------
    # OUTILS
    # -----------------------------------------------------------------
    def wipe(self, *mobs):
        """Fait disparaître proprement les mobjects passés en argument."""
        self.play(*[FadeOut(m) for m in mobs])

    # -----------------------------------------------------------------
    # SÉQUENCE PRINCIPALE
    # -----------------------------------------------------------------
    def construct(self):
        # Réglage caméra initial
        self.set_camera_orientation(phi = -25 * DEGREES, frame_center = [0.5, 0, 0] )

        # Axes 3D communs
        axes = ThreeDAxes(
            x_range=(-4, 4, 2),
            y_range=(-4, 4, 2),
            z_range=(-4, 4, 2),
            x_length=6,
            y_length=6,
            z_length=6,
        )
        self.add(axes)

        # -------------------- SECTION 1 : Sol plan --------------------
        title1 = Text("1 · Sol plan y = 0", font_size=40).to_edge(UP)
        eq1 = MathTex(r"y\;\geqslant\;0,\quad \mathbf n=(0,1,0)^{\top}", font_size=40)
        eq1.next_to(title1, DOWN)

        plane = Square(side_length=8, fill_opacity=0.3, fill_color=BLUE_E)
        plane.rotate(PI / 2, axis=RIGHT)

        self.play(Write(title1), Write(eq1))
        self.play(FadeIn(plane, shift=IN))
        self.wait(2)
        self.wipe(title1, eq1, plane)

        # ---------------- SECTION 2 : Chute libre ---------------------
        title2 = Text("2 · Chute libre – Euler semi-implicite", font_size=40).to_edge(UP)
        eq2 = MathTex(
            r"\mathbf v_{t+\Delta t} = \mathbf v_t + \mathbf g\,\Delta t,\;"
            r"\mathbf x_{t+\Delta t} = \mathbf x_t + \mathbf v_{t+\Delta t}\,\Delta t",
            font_size=36
        ).scale(0.9)
        eq2.next_to(title2, DOWN)

        cube2 = Cube(side_length=1)
        cube2.shift(UP * 3)
        g_vec = Arrow(cube2.get_center(), cube2.get_center() + DOWN * 2, buff=0, color=YELLOW)

        self.play(Write(title2), Write(eq2))
        self.play(FadeIn(cube2), GrowArrow(g_vec))
        self.wait(2)
        self.play(
            cube2.animate.shift(DOWN * 3),
            g_vec.animate.shift(DOWN * 3),
            run_time=2,
            rate_func=linear,
        )
        self.wipe(title2, eq2, cube2, g_vec)

        # -------------- SECTION 3 : Collision & pénétration ----------
        title3 = Text("3 · Collision – Correction de pénétration", font_size=40).to_edge(UP)
        eq3 = MathTex(
            r"\delta = \max(0,\,-p_y),\;"
            r"\mathbf x\gets\mathbf x + \delta\,\mathbf n",
            font_size=40
        )
        eq3.next_to(title3, DOWN)

        cube3 = Cube(side_length=1).shift(UP * 0.6)
        plane3 = plane.copy().set_fill(RED_E, opacity=0.3)

        self.play(Write(title3), Write(eq3))
        self.play(FadeIn(plane3), FadeIn(cube3))
        # collision : descendre puis corriger
        self.play(cube3.animate.shift(DOWN * 1.2), run_time=1)
        self.play(cube3.animate.shift(UP * 0.2), run_time=0.5)
        self.wait(1)
        self.wipe(title3, eq3, cube3, plane3)

        # ---------------- SECTION 4 : Rotation cube ------------------
        title4 = Text("4 · Rotation – θ, ω", font_size=40).to_edge(UP)
        eq4 = MathTex(
            r"\boldsymbol\theta_{t+\Delta t} = "
            r"\boldsymbol\theta_t + \boldsymbol\omega_t\,\Delta t",
            font_size=40
        )
        eq4.next_to(title4, DOWN)

        cube4 = Cube(side_length=1)
        self.play(Write(title4), Write(eq4), FadeIn(cube4))
        self.play(Rotate(cube4, angle=PI / 2, axis=RIGHT), run_time=2)
        self.wait(1)
        self.wipe(title4, eq4, cube4)

        # ------------- SECTION 5 : Joint & articulation --------------
        title5 = Text("5 · Articulation – Angle imposé", font_size=40).to_edge(UP)
        eq5 = MathTex(
            r"\alpha_\text{joint} = \arccos\left("
            r"\frac{(\mathbf u\cdot\mathbf v)}{\|\,\mathbf u\|\|\,\mathbf v\|}"
            r"\right)",
            font_size=40
        )
        eq5.next_to(title5, DOWN)

        cube_a = Cube(side_length=0.8).shift(LEFT * 1.2)
        cube_b = Cube(side_length=0.8).shift(RIGHT * 1.2)
        joint_line = Line(cube_a.get_center(), cube_b.get_center(), color=GREEN)

        self.play(Write(title5), Write(eq5))
        self.play(FadeIn(cube_a), FadeIn(cube_b), GrowFromCenter(joint_line))
        self.play(
            Rotate(cube_b, angle=PI / 4, axis=OUT, about_point=cube_a.get_center()),
            joint_line.animate.put_start_and_end_on(cube_a.get_center(), cube_b.get_center()),
            run_time=2,
        )
        self.wait(1)
        self.wipe(title5, eq5, cube_a, cube_b, joint_line)

        # ------------ SECTION 6 : Traction horizontale ---------------
        title6 = Text("6 · Traction horizontale & friction", font_size=40).to_edge(UP)
        eq6 = MathTex(
            r"\mathbf J_\text{traction} = -\frac{m\,\Delta\mathbf r_{\,\parallel}}{\Delta t},\;"
            r"|\mathbf J|\le \mu_s m g\,\Delta t",
            font_size=36
        ).scale(0.9)
        eq6.next_to(title6, DOWN)

        cube6 = Cube(side_length=1).shift(UP * 0.5 + LEFT * 2)
        traj_arrow = Arrow(
            cube6.get_center(),
            cube6.get_center() + RIGHT * 4,
            buff=0.1,
            color=ORANGE,
            stroke_width=4,
        )

        self.play(Write(title6), Write(eq6))
        self.play(FadeIn(cube6))
        self.play(GrowArrow(traj_arrow), cube6.animate.shift(RIGHT * 4), run_time=2)
        self.wait(1)
        self.wipe(title6, eq6, cube6, traj_arrow)

        # Fin
        end_text = Text("Fin – Merci !", font_size=48)
        self.play(Write(end_text))
        self.wait(2)


def cleanup_temp_folders():
    """Nettoie les dossiers temporaires créés par Manim."""
    temp_folders = ["partial_movie_files", "Tex", "texts"]
    video_output_dir = "./video_output"
    
    for folder in temp_folders:
        folder_path = os.path.join(video_output_dir, folder)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"✓ Dossier temporaire supprimé : {folder}")
            except Exception as e:
                print(f"✗ Erreur lors de la suppression de {folder}: {e}")


# ---------------------------------------------------------------------
# REND AUTOMATIQUEMENT LA VIDÉO
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    config.media_dir = config.video_dir = config.images_dir = "./video_output"
    if os.path.exists("./video_output"):
        shutil.rmtree("./video_output")
    Showcase().render()

    # Nettoyage des dossiers temporaires créés par Manim
    cleanup_temp_folders()

    print("Vidéo sauvegardée → ./video_output/")
