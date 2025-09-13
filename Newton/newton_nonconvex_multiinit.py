from manim import *
import numpy as np

# ---------- 工具函数 ----------
def taylor2_quad(f, df, d2f, x0):
    """返回二阶泰勒近似函数 q(t)"""
    f0 = f(x0)
    g0 = df(x0)
    h0 = d2f(x0)
    def q(t):
        return f0 + g0*(t - x0) + 0.5*h0*(t - x0)**2
    return q

def safe_newton_step(df, d2f, x, eps=1e-8, damping_if_tiny=True):
    """牛顿迭代步，带阻尼处理（仅用于可视化稳定）"""
    g = df(x)
    h = d2f(x)
    if abs(h) < eps:
        step = -g / (np.sign(h) * eps if np.sign(h)!=0 else eps)
        if damping_if_tiny:
            step *= 0.2
        return x + step, True
    return x - g / h, False

# ---------- 公共基类：提供 f/df/d2f 与关键点标注 ----------
class _NewtonNonconvexBase(Scene):
    # 统一函数：f(x)=x^4-3x^2
    def f(self, x):  return x**4 - 3*x**2
    def df(self, x): return 4*x**3 - 6*x
    def d2f(self, x): return 12*x**2 - 6

    def mark_critical_points(self, ax, scale=0.45):
        """标记 x=0(局部最大) 与 ±sqrt(3/2)(局部最小)"""
        xs = [-np.sqrt(3/2), 0.0, np.sqrt(3/2)]
        labels = ["Local min", "Local max", "Local min"]
        colors = [GREEN, RED, GREEN]
        for x, lab, col in zip(xs, labels, colors):
            p = Dot(ax.coords_to_point(x, self.f(x)), color=col, radius=0.06)
            # 改动1：先创建垂直线，再用set方法设置样式（兼容旧版本）
            v = ax.get_vertical_line(ax.coords_to_point(x, self.f(x)))
            v.set_color(col).set_stroke(opacity=0.35)  # 分离样式设置
            t = Text(lab, color=col).scale(scale)
            t.next_to(ax.coords_to_point(x, self.f(x)), UP, buff=0.08)
            self.play(FadeIn(p), Create(v), FadeIn(t), run_time=0.35)

    def make_info_panel(self, anchor_mobj, title="Iteration Info"):
        """右上角信息面板（返回容器与内部行占位）"""
        bg = RoundedRectangle(corner_radius=0.12, width=4.9, height=2.5,
                              fill_opacity=0.12, stroke_opacity=0.3)
        bg.next_to(anchor_mobj, UP, buff=0.2).align_to(anchor_mobj, RIGHT)
        ttl = Text(title, weight=BOLD).scale(0.45).move_to(bg.get_top()).shift(DOWN*0.32)
        eq  = MathTex(r"x_{k+1}=x_k-\frac{f'(x_k)}{f''(x_k)}").scale(0.55).next_to(ttl, DOWN, buff=0.12)
        lines = VGroup(
            MathTex(r"x_k=?").scale(0.55),
            MathTex(r"f'(x_k)=?").scale(0.55),
            MathTex(r"f''(x_k)=?").scale(0.55),
            MathTex(r"\text{Curvature: }?").scale(0.55),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.07).next_to(eq, DOWN, buff=0.15).align_to(eq, LEFT)
        panel = VGroup(bg, ttl, eq, lines)
        panel.scale_to_fit_width(5.1)
        return panel, lines

    def draw_function(self, ax):
        g = ax.plot(lambda t: self.f(t), color=YELLOW,
                    x_range=(ax.x_range[0], ax.x_range[1]))
        lab = MathTex("f(x)").scale(0.55).next_to(g, UR, buff=0.12)
        self.play(Create(ax), run_time=0.5)
        self.play(Create(g), FadeIn(lab), run_time=0.6)
        return g, lab

    def one_side_newton(
        self, ax, x0, max_iters=6,
        color_curr=RED, color_next=GREEN, color_quad=BLUE,
        info_title="Iteration Info", show_grid=True
    ):
        """在一个坐标轴上执行并可视化牛顿迭代"""
        # 背景网格（只覆盖该轴区域）
        if show_grid:
            grid = NumberPlane(
                x_range=ax.x_range, y_range=ax.y_range,
                background_line_style={"stroke_opacity": 0.12}
            ).match_width(ax).match_height(ax).move_to(ax)
            self.add(grid)

        # 函数曲线 & 标注
        f_graph, f_label = self.draw_function(ax)
        self.mark_critical_points(ax, scale=0.42)

        # 信息面板（对齐到该轴右上）
        panel, info_lines = self.make_info_panel(ax, title=info_title)
        self.add(panel)

        # 初始点
        xk = x0
        x_dot = Dot(ax.coords_to_point(xk, self.f(xk)), color=color_curr, radius=0.08)
        # 改动2：初始点垂直线，同样先创建再设置样式
        x_vline = ax.get_vertical_line(ax.coords_to_point(xk, self.f(xk)))
        x_vline.set_color(color_curr).set_stroke(opacity=0.7, width=2)  # 分离样式设置
        x_label = MathTex("x_0").scale(0.6).next_to(x_vline, DOWN, buff=0.08)
        self.play(FadeIn(x_dot), Create(x_vline), FadeIn(x_label), run_time=0.5)

        quad_graph = None
        xk_marker = VGroup(x_dot, x_vline, x_label)

        for k in range(max_iters):
            gk = self.df(xk)
            hk = self.d2f(xk)
            curvature_text = r"\text{Convex (min)}" if hk > 0 else r"\text{Concave (max)}" if hk < 0 else r"\text{Flat}"

            new_info = VGroup(
                MathTex(rf"x_k={xk:.6f}").scale(0.55),
                MathTex(rf"f'(x_k)={gk:.4e}").scale(0.55),
                MathTex(rf"f''(x_k)={hk:.4e}").scale(0.55),
                MathTex(curvature_text).scale(0.55).set_color(GREEN if hk>0 else RED if hk<0 else GRAY)
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.07).move_to(info_lines, aligned_edge=LEFT)
            self.play(Transform(info_lines, new_info), run_time=0.4)

            # 二次近似 q_k
            qk = taylor2_quad(self.f, self.df, self.d2f, xk)
            local_w = 1.8
            xa, xb = max(ax.x_range[0], xk - local_w), min(ax.x_range[1], xk + local_w)
            new_quad_graph = ax.plot(lambda t: qk(t), color=color_quad, x_range=(xa, xb), stroke_width=2)

            quad_label = MathTex(r"q_k(t)").scale(0.6).set_color(color_quad)
            quad_label.next_to(ax.coords_to_point(xb, qk(xb)), UR, buff=0.08)
            quad_label.add_background_rectangle(buff=0.03, opacity=0.6)

            if quad_graph is None:
                self.play(Create(new_quad_graph), FadeIn(quad_label), run_time=0.5)
            else:
                self.play(ReplacementTransform(quad_graph, new_quad_graph),
                          FadeTransform(quad_label_prev, quad_label), run_time=0.5)
            quad_graph = new_quad_graph
            quad_label_prev = quad_label

            # 新点
            x_next, used_damping = safe_newton_step(self.df, self.d2f, xk)
            x_next_dot = Dot(ax.coords_to_point(x_next, self.f(x_next)), color=color_next, radius=0.08)
            # 改动3：新点垂直线，先创建再设置样式
            x_next_vline = ax.get_vertical_line(ax.coords_to_point(x_next, self.f(x_next)))
            x_next_vline.set_color(color_next).set_stroke(opacity=0.7, width=2)  # 分离样式设置
            x_next_label = MathTex(fr"x_{{{k+1}}}").scale(0.6).set_color(color_next).next_to(x_next_vline, DOWN, buff=0.08)

            # 底部箭头显示 Δx
            step_arrow = Arrow(
                ax.coords_to_point(xk, ax.y_range[0]+0.4),
                ax.coords_to_point(x_next, ax.y_range[0]+0.4),
                color=YELLOW, stroke_width=2, tip_length=0.14
            )
            step_label = MathTex(fr"\Delta x={x_next-xk:.4f}").scale(0.5).set_color(YELLOW)
            step_label.next_to(step_arrow, UP, buff=0.05)

            # 若阻尼，给个小提示（靠近面板底部，避免遮挡）
            damp_tip = None
            if used_damping:
                damp_tip = Text("curvature≈0 → damped", slant=ITALIC).scale(0.42).set_color(ORANGE)
                damp_tip.next_to(panel, DOWN, buff=0.08).align_to(panel, RIGHT)
                damp_tip.add_background_rectangle(opacity=0.55, buff=0.02)
                self.play(FadeIn(damp_tip), run_time=0.25)

            self.play(FadeIn(x_next_dot), Create(x_next_vline), FadeIn(x_next_label),
                      Create(step_arrow), FadeIn(step_label), run_time=0.6)

            # 历史点弱化
            self.play(xk_marker.animate.set_opacity(0.2), run_time=0.25)

            # 新点→当前点
            xk_marker = VGroup(
                x_next_dot.copy().set_color(color_curr),
                x_next_vline.copy().set_color(color_curr),
                MathTex(fr"x_{{{k+1}}}").scale(0.6).next_to(x_next_vline, DOWN, buff=0.08).set_color(color_curr)
            )
            self.play(
                ReplacementTransform(x_next_dot, xk_marker[0]),
                ReplacementTransform(x_next_vline, xk_marker[1]),
                ReplacementTransform(x_next_label, xk_marker[2]),
                FadeOut(step_arrow),
                FadeOut(step_label),
                run_time=0.45
            )

            if damp_tip:
                self.play(FadeOut(damp_tip), run_time=0.2)

            xk = x_next

        # 收尾：判别性质并提示
        is_max = self.d2f(xk) < 0
        result_text = Text(
            f"Converged to a {'local maximum' if is_max else 'local minimum'}",
            color=RED if is_max else GREEN
        ).scale(0.5)
        result_text.next_to(ax, DOWN, buff=0.1)
        result_text.add_background_rectangle(opacity=0.65)
        self.play(FadeIn(result_text), run_time=0.6)

# ---------- 新的对比场景 ----------
class NewtonNonconvex_MultiInit_Compare(_NewtonNonconvexBase):
    """
    左/右两个面板对比不同初值的收敛结果：
      - 左：x0=0.2 → 往往收敛到局部最大 x=0
      - 右：x0=1.5 → 收敛到右侧局部最小 x=+sqrt(3/2)
    你也可以把右侧的 x0 改为 -1.5，看收敛到左侧最小值。
    """
    # 坐标范围设置在每个面板内保持一致，避免比例失真
    x_range = (-2.8, 2.8, 1)
    y_range = (-3.5, 5.5, 1)
    max_iters = 7

    def construct(self):
        # 顶部总标题
        title = Text("Newton on Nonconvex f(x)=x^4-3x^2 — Initial Value Matters",
                     weight=BOLD).scale(0.6).to_edge(UP, buff=0.25)
        self.add(title)

        # 左右两个坐标轴
        ax_left = Axes(
            x_range=self.x_range, y_range=self.y_range,
            x_length=6.0, y_length=3.8,
            axis_config={"include_numbers": True, "font_size": 22}, tips=False
        )
        ax_right = ax_left.copy()

        # 布局：左右并排，底部对齐
        group = VGroup(ax_left, ax_right).arrange(RIGHT, buff=1.2).to_edge(DOWN, buff=0.6)
        ax_left, ax_right = group[0], group[1]  # 更新变换后实例

        # # 每侧小标题
        # cap_left  = Text("Init: x0 = 0.2", color=RED).scale(0.5)
        # cap_right = Text("Init: x0 = 1.5", color=GREEN).scale(0.5)
        # cap_left.next_to(ax_left, UP, buff=0.15)
        # cap_right.next_to(ax_right, UP, buff=0.15)
        # self.add(cap_left, cap_right)

        # 左侧执行（初值靠近 0）
        self.one_side_newton(
            ax=ax_left, x0=0.2, max_iters=self.max_iters,
            color_curr=RED, color_next=GREEN, color_quad=BLUE,
            info_title="Left Panel — Iteration", show_grid=True
        )

        # 右侧执行（初值靠近右最小值）
        self.one_side_newton(
            ax=ax_right, x0=1.5, max_iters=self.max_iters,
            color_curr=MAROON, color_next=TEAL, color_quad=BLUE_E,
            info_title="Right Panel — Iteration", show_grid=True
        )

        self.wait(1.2)

# 单面板版本（保留）
class NewtonNonconvex_MultiStationary(_NewtonNonconvexBase):
    """保留单轴版本（单独演示某一个初值）"""
    x0 = 0.2
    def construct(self):
        title = Text("Newton (Single Panel)  f(x)=x^4-3x^2", weight=BOLD).scale(0.6).to_edge(UP, buff=0.25)
        self.add(title)
        ax = Axes(
            x_range=(-2.8, 2.8, 1), y_range=(-3.5, 5.5, 1),
            x_length=10, y_length=5.2,
            axis_config={"include_numbers": True, "font_size": 24}, tips=False
        ).to_edge(DOWN, buff=0.5)
        self.one_side_newton(
            ax=ax, x0=self.x0, max_iters=7,
            color_curr=RED, color_next=GREEN, color_quad=BLUE,
            info_title="Iteration Info (Single Panel)", show_grid=True
        )

if __name__ == "__main__":
    from manim import config
    config.media_dir = "./media"
    # 预览双面板对比（运行此场景）
    scene = NewtonNonconvex_MultiInit_Compare()
    scene.render()