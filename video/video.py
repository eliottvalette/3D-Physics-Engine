from manim import *
import numpy as np
import shutil


class Showcase(Scene):
    """Vidéo pédagogique : 1 graphique + 1 équation \npar *section* (5 sections)."""

    def construct(self):
        # ------------------------------------------------------------
        # 0 · Titre
        # ------------------------------------------------------------
        title = Text("Mini-Moteur Physique 3D : Panorama", font_size=60)
        subtitle = Text("Cube → Quadrupède articulé", font_size=36)
        subtitle.next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(title, shift=UP), FadeIn(subtitle))
        self.wait(1.4)
        self.play(FadeOut(title), FadeOut(subtitle))

        # Helper to fade / remove previous mobjects
        def wipe(*mobs):
            self.play(*[FadeOut(m) for m in mobs])

        # ------------------------------------------------------------
        # 1 · Sol
        # ------------------------------------------------------------
        h1 = Text("1  ·  Demi-espace  y ≥ 0").to_edge(UP)
        eq_plane = Text("\mathbf n\cdot(\mathbf p-\mathbf p_0)=0,\;\mathbf n=(0,1,0)")
        graph1_axes = Axes(x_range=[0,1,0.2], y_range=[0,50,10], x_length=5, y_length=2.7)
        t = np.linspace(0,1,100); g=9.81; Ep= g*(1-0.5*g*t**2)
        curve1 = graph1_axes.plot(lambda x: g*(1-0.5*g*x**2), x_range=[0,1])
        graph1_label = graph1_axes.get_axis_labels(Text("t (s)"), Text("E_p (J)"))

        self.play(Write(h1))
        self.play(Write(eq_plane))
        self.wait(0.6)
        self.play(Create(graph1_axes), Write(graph1_label), Create(curve1))
        self.wait(1.2)
        wipe(h1, eq_plane, graph1_axes, graph1_label, curve1)

        # ------------------------------------------------------------
        # 2 · Chute d’un cube
        # ------------------------------------------------------------
        h2 = Text("2  ·  Chute & impact").to_edge(UP)
        eq_euler = Text("∑1/n^2 = 2/π")
        axes2 = Axes([0,1.1,0.2], [0,5,1], x_length=5, y_length=2.7)
        y_curve = axes2.plot(lambda x: 5-0.5*9.81*x**2, x_range=[0,1])
        axes2_labels = axes2.get_axis_labels(Text("t"), Text("y"))

        self.play(Write(h2))
        self.play(Write(eq_euler))
        self.wait(0.6)
        self.play(Create(axes2), Write(axes2_labels), Create(y_curve))
        self.wait(1.2)
        wipe(h2, eq_euler, axes2, axes2_labels, y_curve)

        # ------------------------------------------------------------
        # 3 · Rebond sur un angle
        # ------------------------------------------------------------
        h3 = Text("3  ·  Rebond & couple").to_edge(UP)
        eq_impulse = Text("∑1/n^2 = 2/π")
        axes3 = Axes([0,0.6,0.1], [-2,32,10], x_length=5, y_length=2.7)
        ωx = lambda t: 30*np.exp(-5*t)
        ωz = lambda t: 18*np.exp(-5*t)
        curve_ωx = axes3.plot(ωx, x_range=[0,0.6]); curve_ωz = axes3.plot(ωz, x_range=[0,0.6], color=YELLOW)
        axes3_labels = axes3.get_axis_labels(Text("t"), Text("ω (rad/s)"))
        ω_legend = VGroup(Text("ω_x", color=WHITE), Text("ω_z", color=YELLOW)).arrange(DOWN, aligned_edge=LEFT).scale(0.4)
        ω_legend.next_to(axes3, RIGHT)

        self.play(Write(h3))
        self.play(Write(eq_impulse))
        self.wait(0.6)
        self.play(Create(axes3), Write(axes3_labels), Create(curve_ωx), Create(curve_ωz), FadeIn(ω_legend))
        self.wait(1.2)
        wipe(h3, eq_impulse, axes3, axes3_labels, curve_ωx, curve_ωz, ω_legend)

        # ------------------------------------------------------------
        # 4 · Inertie composée
        # ------------------------------------------------------------
        h4 = Text("4  ·  Inertie composée").to_edge(UP)
        eq_inertia = Text("\mathbf I_{tot}=\sum m_i(\mathbf I_i + [\![\mathbf d_i]\!])")
        matrix = Text("\mathbf I_{tot}=\begin{bmatrix} 0.12 & 0 & 0 \\ 0 & 0.10 & 0 \\ 0 & 0 & 0.14 \end{bmatrix}").scale(0.75)
        matrix.next_to(eq_inertia, DOWN)

        self.play(Write(h4))
        self.play(Write(eq_inertia))
        self.play(Write(matrix))
        self.wait(1.2)
        wipe(h4, eq_inertia, matrix)

        # ------------------------------------------------------------
        # 5 · Traction & friction
        # ------------------------------------------------------------
        h5 = Text("5  ·  Traction : μ_s / μ_k").to_edge(UP)
        eq_friction = Text("|\mathbf J_t| \le \mu_s |\mathbf J_n|")
        axes5 = Axes([-0.3,0.3,0.1], [0,0.6,0.2], x_length=5, y_length=2.7)
        def μ_curve(x):
            μ_s, μ_k, v_s = 0.5, 0.25, 0.02
            return μ_s*0.3 if abs(x)<v_s else μ_k*0.3
        fcurve = axes5.plot(μ_curve, x_range=[-0.3,0.3], use_smoothing=False)
        axes5_labels = axes5.get_axis_labels(Text("v_t"), Text("|J_t|"))

        self.play(Write(h5))
        self.play(Write(eq_friction))
        self.play(Create(axes5), Write(axes5_labels), Create(fcurve))
        self.wait(1.2)
        wipe(h5, eq_friction, axes5, axes5_labels, fcurve)

        # ------------------------------------------------------------
        # Finale : mosaïque des 5 titres
        # ------------------------------------------------------------
        titles = VGroup(*[Text(txt, font_size=28) for txt in [
            "Sol", "Chute", "Rebond", "Inertie", "Friction"
        ]]).arrange(DOWN, buff=0.5)
        self.play(FadeIn(titles))
        self.wait(2)


# ------------------------------------------------------------
# Rend automatiquement la vidéo
# ------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    config.media_dir = config.video_dir = config.images_dir = "./video_output"
    if os.path.exists("./video_output"):
        shutil.rmtree("./video_output")
    Showcase().render()
    print("Vidéo sauvegardée → ./video_output/")