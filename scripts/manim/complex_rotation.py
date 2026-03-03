"""
Manim Animation: Complex Rotation in Kähler Structure

This script creates high-quality mathematical animations using Manim
to visualize complex plane rotation and Kähler structure concepts.

Requirements:
    pip install manim
    MiKTeX (LaTeX distribution for Windows)

Usage:
    manim -pql complex_rotation.py ComplexRotation
    manim -pql complex_rotation.py KaehlerStructure
"""

import os
import sys

# Configure MiKTeX path for Windows
miktex_path = r"C:\Program Files\MiKTeX\miktex\bin\x64"
if os.path.exists(miktex_path):
    os.environ["PATH"] = miktex_path + os.pathsep + os.environ.get("PATH", "")

from manim import *
from manim.utils.tex import TexTemplate
import numpy as np

# Create custom tex template with MiKTeX
custom_template = TexTemplate()
custom_template.tex_compiler = "latex"
if os.path.exists(miktex_path):
    # Use full path to latex compiler on Windows
    latex_exe = os.path.join(miktex_path, "latex.exe")
    if os.path.exists(latex_exe):
        custom_template.tex_compiler = latex_exe

# Set as default template
config["text"]["tex_template"] = custom_template


class ComplexRotation(Scene):
    """
    Animation showing a point rotating on the unit circle in complex plane.
    
    Demonstrates z(t) = e^(i·ω·t) = cos(ω·t) + i·sin(ω·t)
    """
    
    def construct(self):
        # Title
        title = MathTex(r"z(t) = e^{i\omega t} = \cos(\omega t) + i\sin(\omega t)")
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Create complex plane
        plane = ComplexPlane(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        plane.add_coordinates()
        
        # Labels
        real_label = MathTex(r"\text{Real: } \cos(\omega t)").next_to(plane, DOWN, buff=0.5)
        imag_label = MathTex(r"\text{Imag: } \sin(\omega t)").next_to(plane, RIGHT, buff=0.5)
        
        self.play(Create(plane), run_time=2)
        self.play(Write(real_label), Write(imag_label))
        self.wait(0.5)
        
        # Unit circle
        circle = Circle(radius=1, color=YELLOW, stroke_width=2)
        self.play(Create(circle), run_time=1)
        
        # Create rotating point
        omega = 1.0  # Angular velocity
        
        # Initial position (t=0)
        dot = Dot(point=plane.c2p(1, 0), color=RED, radius=0.1)
        radius_line = Line(plane.c2p(0, 0), dot.get_center(), color=WHITE, stroke_width=2)
        
        # Angle arc
        angle_tracker = ValueTracker(0)
        angle_arc = always_redraw(lambda: Arc(
            radius=0.3,
            start_angle=0,
            angle=angle_tracker.get_value(),
            color=GREEN
        ))
        
        # Angle label
        angle_label = always_redraw(lambda: MathTex(
            f"{angle_tracker.get_value():.2f}"
        ).next_to(angle_arc, UR, buff=0.1).scale(0.7))
        
        self.play(
            Create(dot),
            Create(radius_line),
            Create(angle_arc),
            Write(angle_label)
        )
        
        # Animate rotation
        self.play(
            angle_tracker.animate.set_value(2 * PI),
            Rotate(dot, angle=2 * PI, about_point=plane.c2p(0, 0), rate_func=linear),
            Rotate(radius_line, angle=2 * PI, about_point=plane.c2p(0, 0), rate_func=linear),
            run_time=6,
            rate_func=linear
        )
        
        # Trail effect (create path)
        trail = TracedPath(dot.get_center, stroke_color=RED, stroke_width=3, dissipating_time=2)
        self.add(trail)
        
        # Continue rotation with trail
        self.play(
            angle_tracker.animate.set_value(4 * PI),
            Rotate(dot, angle=2 * PI, about_point=plane.c2p(0, 0), rate_func=linear),
            Rotate(radius_line, angle=2 * PI, about_point=plane.c2p(0, 0), rate_func=linear),
            run_time=4,
            rate_func=linear
        )
        
        self.wait(1)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(plane),
            FadeOut(real_label),
            FadeOut(imag_label),
            FadeOut(circle),
            FadeOut(dot),
            FadeOut(radius_line),
            FadeOut(angle_arc),
            FadeOut(angle_label),
            FadeOut(trail)
        )


class KaehlerStructure(Scene):
    """
    Animation demonstrating Kähler structure with multiple frequency components.
    
    Shows how different frequencies create different rotation speeds on
the complex plane, forming the Kähler manifold structure.
    """
    
    def construct(self):
        # Title
        title = Text("Kähler Structure: Multiple Frequency Components", font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        
        # Create complex plane
        plane = ComplexPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        )
        plane.scale(0.8)
        plane.shift(DOWN * 0.5)
        
        self.play(Create(plane), run_time=2)
        
        # Multiple frequencies
        frequencies = [
            (0.5, BLUE, r"\omega_1 = 0.5"),
            (1.0, GREEN, r"\omega_2 = 1.0"),
            (2.0, RED, r"\omega_3 = 2.0")
        ]
        
        dots = []
        lines = []
        labels = []
        
        for i, (omega, color, label_str) in enumerate(frequencies):
            # Create dot
            angle = 0
            x = np.cos(angle)
            y = np.sin(angle)
            dot = Dot(point=plane.c2p(x, y), color=color, radius=0.08)
            
            # Create radius line
            line = Line(plane.c2p(0, 0), dot.get_center(), color=color, stroke_width=2)
            
            # Label
            label = MathTex(label_str, color=color).scale(0.7)
            label.next_to(plane, RIGHT, buff=0.3)
            label.shift(UP * (1 - i * 0.6))
            
            dots.append(dot)
            lines.append(line)
            labels.append(label)
            
            self.play(Create(dot), Create(line), Write(label), run_time=0.5)
        
        self.wait(0.5)
        
        # Animate all rotating at different speeds
        rotations = [
            (dots[0], lines[0], frequencies[0][0]),
            (dots[1], lines[1], frequencies[1][0]),
            (dots[2], lines[2], frequencies[2][0])
        ]
        
        # Create animations
        anims = []
        for dot, line, omega in rotations:
            angle = 2 * PI * omega
            anims.append(Rotate(dot, angle=angle, about_point=plane.c2p(0, 0), 
                               rate_func=linear, run_time=6))
            anims.append(Rotate(line, angle=angle, about_point=plane.c2p(0, 0), 
                             rate_func=linear, run_time=6))
        
        self.play(*anims, run_time=6)
        
        # Add explanatory text
        explanation = Text(
            "Higher frequency → Faster rotation → Higher spatial resolution",
            font_size=24,
            color=YELLOW
        )
        explanation.next_to(plane, DOWN, buff=0.8)
        
        self.play(Write(explanation))
        self.wait(2)
        
        # Fade out
        self.play(
            *[FadeOut(mob) for mob in [title, plane, explanation] + dots + lines + labels]
        )


class FrequencyDistribution(Scene):
    """
    Animation showing how frequency distribution affects kernel behavior.
    """
    
    def construct(self):
        # Title
        title = Text("Frequency Distribution Impact", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Axes for frequency plot
        axes = Axes(
            x_range=[0, 32, 8],
            y_range=[0, 1, 0.25],
            axis_config={"color": WHITE},
            x_axis_config={"numbers_to_include": range(0, 33, 8)},
            y_axis_config={"numbers_to_include": [0, 0.25, 0.5, 0.75, 1]}
        )
        axes.shift(DOWN * 0.5)
        
        x_label = MathTex(r"k \text{ (frequency index)}", font_size=28).next_to(axes, DOWN, buff=0.3)
        y_label = MathTex(r"|\omega_k|", font_size=28).next_to(axes, LEFT, buff=0.3).rotate(90 * DEGREES)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Different frequency distributions
        dim = 32
        k = np.arange(dim // 2)
        
        # TCFMamba: a=3
        freqs_tcf = (-k / dim) ** 3
        freqs_tcf = np.abs(freqs_tcf) / np.max(freqs_tcf)  # Normalize
        
        # Transformer: geometric (approximated)
        freqs_trans = np.exp(-0.1 * k)
        freqs_trans = freqs_trans / np.max(freqs_trans)
        
        # Plot TCFMamba
        points_tcf = [axes.c2p(x, y) for x, y in zip(k, freqs_tcf)]
        graph_tcf = VMobject()
        graph_tcf.set_points_as_corners(points_tcf)
        graph_tcf.set_color(BLUE)
        graph_tcf.set_stroke(width=3)
        
        label_tcf = Text("TCFMamba (a=3)", font_size=24, color=BLUE)
        label_tcf.next_to(axes, RIGHT, buff=0.5).shift(UP * 1.5)
        
        self.play(Create(graph_tcf), Write(label_tcf), run_time=2)
        self.wait(0.5)
        
        # Plot Transformer
        points_trans = [axes.c2p(x, y) for x, y in zip(k, freqs_trans)]
        graph_trans = VMobject()
        graph_trans.set_points_as_corners(points_trans)
        graph_trans.set_color(RED)
        graph_trans.set_stroke(width=3)
        
        label_trans = Text("Transformer (geometric)", font_size=24, color=RED)
        label_trans.next_to(axes, RIGHT, buff=0.5).shift(UP * 0.5)
        
        self.play(Create(graph_trans), Write(label_trans), run_time=2)
        self.wait(1)
        
        # Highlight difference
        brace = BraceBetweenPoints(axes.c2p(0, 0.5), axes.c2p(8, 0.5), direction=UP)
        brace_text = Text("Low-freq dense", font_size=20).next_to(brace, UP, buff=0.1)
        
        self.play(Create(brace), Write(brace_text))
        self.wait(2)
        
        # Fade out
        self.play(
            FadeOut(title),
            FadeOut(axes),
            FadeOut(x_label),
            FadeOut(y_label),
            FadeOut(graph_tcf),
            FadeOut(graph_trans),
            FadeOut(label_tcf),
            FadeOut(label_trans),
            FadeOut(brace),
            FadeOut(brace_text)
        )


if __name__ == "__main__":
    # Render settings
    config.pixel_height = 720
    config.pixel_width = 1280
    config.frame_rate = 30
