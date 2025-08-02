from manim import *
import numpy as np
import shutil


class Showcase(ThreeDScene):
    """Vidéo pédagogique : 1 visuel + 1 équation par section (6 sections)."""

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
        # --------------------------------------------------------------
        # 0 · TITRE
        # --------------------------------------------------------------
        title = Text("Mini-Moteur Physique 3D : Panorama", font_size=60)
        subtitle = Text("Cube → Quadrupède articulé", font_size=36)
        subtitle.next_to(title, DOWN, buff=0.4)

        self.add_fixed_orientation_mobjects(title, subtitle)
        self.play(FadeIn(title, shift=UP), FadeIn(subtitle))
        self.wait(1.4)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ==============================================================
        # 1 · SOL : DEMI-ESPACE y ≥ 0
        # ==============================================================
        self.set_camera_orientation(phi=70 * DEGREES,
                                    theta=-45 * DEGREES,
                                    focal_distance=400)

        h1 = Text("1 · Sol (y ≥ 0)").to_edge(UP)
        eq_plane = Text("Condition :  y ≥ 0     n = (0, 1, 0)")

        self.add_fixed_orientation_mobjects(h1, eq_plane)
        self.play(Write(h1), Write(eq_plane))

        axes3d = ThreeDAxes(x_length=4, y_length=3, z_length=4)

        # Square dans le plan XZ (y = 0) → plan horizontal
        ground = Square(side_length=4,
                        fill_opacity=0.35,
                        fill_color=BLUE_E,
                        stroke_color=BLUE_E)
        # Par défaut la face est dans le plan XY ; on la fait pivoter
        ground.rotate(-PI / 2, axis=RIGHT)       # vers plan XZ
        ground.shift(DOWN * 0)                   # centré sur y = 0

        # Flèche de la normale n = (0,1,0)
        normal_vec = Arrow(start=[0, 0, 0],
                           end=[0, 1.5, 0],
                           color=RED,
                           buff=0)
        normal_label = Text("n").next_to(normal_vec.get_end(),
                                         RIGHT,
                                         buff=0.1)
        self.add_fixed_orientation_mobjects(normal_label)

        self.play(Create(axes3d), FadeIn(ground))
        self.play(GrowArrow(normal_vec), FadeIn(normal_label))
        self.wait(1.4)
        self.wipe(h1, eq_plane, axes3d, ground, normal_vec,
                  normal_label)

        # ==============================================================
        # 2 · CHUTE D'UN CUBE : INTÉGRATION D'EULER
        # ==============================================================
        self.set_camera_orientation(phi=70 * DEGREES,
                                    theta=-45 * DEGREES,
                                    focal_distance=400)

        h2 = Text("2 · Chute & impact").to_edge(UP)
        eq_euler = Text("v_new = v_old + g·dt      "
                        "x_new = x_old + v_new·dt")
        self.add_fixed_orientation_mobjects(h2, eq_euler)
        self.play(Write(h2), Write(eq_euler))

        axes_drop = ThreeDAxes(x_length=4, y_length=4, z_length=4)

        # On réutilise le même sol pour le visuel
        drop_ground = ground.copy()

        cube = Cube(side_length=1)
        cube.move_to(axes_drop.c2p(0, 3, 0))

        self.play(Create(axes_drop), FadeIn(drop_ground), FadeIn(cube))

        # Animation : chute puis rebond amorti (coefficient 0.7)
        self.play(cube.animate.move_to(axes_drop.c2p(0, 0.5, 0)),
                  run_time=1.2,
                  rate_func=smooth)
        self.play(cube.animate.move_to(axes_drop.c2p(0, 1.1, 0)),
                  run_time=0.6,
                  rate_func=there_and_back)

        self.wait(1.4)
        self.wipe(h2, eq_euler, axes_drop, drop_ground, cube)

        # --------------------------------------------------------------
        # 3 · REBOND & COUPLE
        # --------------------------------------------------------------
        h3 = Text("3 · Rebond & couple").to_edge(UP)
        eq_impulse = Text("J = -v_rel / (1/m + ((r×n)·(r×n)) / I_yy)")
        self.add_fixed_orientation_mobjects(h3, eq_impulse)
        self.play(Write(h3), Write(eq_impulse))

        axes3 = Axes(x_range=[0, 0.6, 0.1],
                     y_range=[-2, 32, 10],
                     x_length=5,
                     y_length=2.7)
        ωx = lambda t: 30 * np.exp(-5 * t)
        ωz = lambda t: 18 * np.exp(-5 * t)
        curve_ωx = axes3.plot(ωx, x_range=[0, 0.6])
        curve_ωz = axes3.plot(ωz, x_range=[0, 0.6], color=YELLOW)
        axes3_lab = axes3.get_axis_labels(Text("t"),
                                          Text("ω (rad/s)"))
        legend = VGroup(Text("ω_x", color=WHITE),
                        Text("ω_z", color=YELLOW)).arrange(DOWN,
                                                           aligned_edge=LEFT
                                                           ).scale(0.4)
        legend.next_to(axes3, RIGHT)

        self.play(Create(axes3), Write(axes3_lab), Create(curve_ωx),
                  Create(curve_ωz), FadeIn(legend))
        self.wait(1.2)
        self.wipe(h3, eq_impulse, axes3, axes3_lab, curve_ωx,
                  curve_ωz, legend)

        # --------------------------------------------------------------
        # 4 · INERTIE COMPOSÉE
        # --------------------------------------------------------------
        h4 = Text("4 · Inertie composée").to_edge(UP)
        eq_inertia = Text("I_tot = Σ m_i ( I_i + d_i² )")
        matrix = Text("I_tot ≈ diag(0.12, 0.10, 0.14)").scale(0.75)
        matrix.next_to(eq_inertia, DOWN)
        self.add_fixed_orientation_mobjects(h4, eq_inertia, matrix)
        self.play(Write(h4), Write(eq_inertia), Write(matrix))
        self.wait(1.2)
        self.wipe(h4, eq_inertia, matrix)

        # --------------------------------------------------------------
        # 5 · TRACTION & FRICTION
        # --------------------------------------------------------------
        h5 = Text("5 · Traction : μ_s / μ_k").to_edge(UP)
        eq_friction = Text("|J_t| ≤ μ_s · |J_n|")
        self.add_fixed_orientation_mobjects(h5, eq_friction)
        self.play(Write(h5), Write(eq_friction))

        axes5 = Axes(x_range=[-0.3, 0.3, 0.1],
                     y_range=[0, 0.6, 0.2],
                     x_length=5,
                     y_length=2.7)

        def μ_curve(x):
            μ_s, μ_k, v_s = 0.5, 0.25, 0.02
            return μ_s * 0.3 if abs(x) < v_s else μ_k * 0.3

        fcurve = axes5.plot(μ_curve, x_range=[-0.3, 0.3],
                            use_smoothing=False)
        axes5_lab = axes5.get_axis_labels(Text("v_t"),
                                          Text("|J_t|"))

        self.play(Create(axes5), Write(axes5_lab), Create(fcurve))
        self.wait(1.2)
        self.wipe(h5, eq_friction, axes5, axes5_lab, fcurve)

        # --------------------------------------------------------------
        # 6 · ARTICULATION ÉPAULE — COUDE
        # --------------------------------------------------------------
        h6 = Text("6 · Articulation : épaule / coude").to_edge(UP)
        eq_joint = Text("θ_new = θ_old + ω · dt")
        self.add_fixed_orientation_mobjects(h6, eq_joint)
        self.play(Write(h6), Write(eq_joint))

        axes6 = Axes(x_range=[0, 1, 0.2],
                     y_range=[-30, 60, 15],
                     x_length=5,
                     y_length=2.7)
        θ_shoulder = lambda t: 45 * np.sin(2 * np.pi * t)
        curve_sh = axes6.plot(θ_shoulder, x_range=[0, 1])
        axes6_lab = axes6.get_axis_labels(Text("t"),
                                          Text("θ (°)"))
        self.play(Create(axes6), Write(axes6_lab), Create(curve_sh))
        self.wait(1.2)
        self.wipe(h6, eq_joint, axes6, axes6_lab, curve_sh)

        # --------------------------------------------------------------
        # FINALE : MOSAÏQUE DES TITRES
        # --------------------------------------------------------------
        titles = VGroup(*[Text(txt, font_size=28) for txt in [
            "Sol", "Chute", "Rebond", "Inertie",
            "Friction", "Articulation"
        ]]).arrange(DOWN, buff=0.5)
        self.add_fixed_orientation_mobjects(titles)
        self.play(FadeIn(titles))
        self.wait(2)


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
    print("Vidéo sauvegardée → ./video_output/")
