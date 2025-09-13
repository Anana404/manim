from manim import *
import numpy as np

# ---------- 通用的工具函数 ----------
def taylor2_quad(f, df, d2f, x0):
    """返回二阶泰勒近似函数 q(t)"""
    f0 = f(x0)
    g0 = df(x0)
    h0 = d2f(x0)
    def q(t):
        return f0 + g0*(t - x0) + 0.5*h0*(t - x0)**2
    return q

def safe_newton_step(df, d2f, x, eps=1e-8, damping_if_tiny=True):
    """牛顿迭代步，带阻尼处理避免数值问题"""
    g = df(x)
    h = d2f(x)
    if abs(h) < eps:
        step = -g / (np.sign(h) * eps if np.sign(h)!=0 else eps)
        if damping_if_tiny:
            step *= 0.2  # 简单阻尼
        return x + step, True
    return x - g / h, False

class BaseNewton1D(Scene):
    """牛顿法可视化基类"""
    x0 = 2.0
    x_range = (-4, 4, 1)
    y_range = (-5, 10, 1)
    max_iters = 8  # 增加迭代次数以展示缓慢收敛
    title_text = "Newton's Method (Optimization)"
    subtitle_text = ""
    show_grid = True
    animation_speed = 1.2  # 放慢动画速度，更清晰展示每一步
    
    def f(self, x):
        return x**2

    def df(self, x):
        return 2*x

    def d2f(self, x):
        return 2.0

    def construct(self):
        # 坐标轴
        ax = Axes(
            x_range=self.x_range,
            y_range=self.y_range,
            x_length=10,
            y_length=6,
            axis_config={"include_numbers": True, "font_size": 28},
            tips=False
        )
        ax.to_edge(DOWN)
        if self.show_grid:
            grid = NumberPlane(
                x_range=self.x_range, y_range=self.y_range,
                background_line_style={"stroke_opacity": 0.15}
            ).match_width(ax).match_height(ax).move_to(ax)
            self.add(grid)

        # 标题
        title = Text(self.title_text, weight=BOLD).scale(0.8).to_edge(UP, buff=0.3).shift(LEFT*2)
        subtitle = Text(self.subtitle_text).scale(0.6).next_to(title, RIGHT, buff=0.5).shift(RIGHT*0.2)
        header = VGroup(title, subtitle)
        header.scale_to_fit_width(min(11.0, config.frame_width-0.8))
        self.add(header)

        # 绘制原函数
        f_graph = ax.plot(lambda t: self.f(t), color=YELLOW, x_range=(self.x_range[0], self.x_range[1]))
        f_label = MathTex("f(x)").scale(0.7).next_to(f_graph, UR, buff=0.2)
        self.play(Create(ax), run_time=0.8)
        self.play(Create(f_graph), FadeIn(f_label), run_time=1.0)

        # 迭代信息面板
        panel_bg = RoundedRectangle(corner_radius=0.15, width=5.6, height=2.6, fill_opacity=0.12, stroke_opacity=0.3)
        panel_bg.to_corner(UR, buff=0.25)
        panel_title = Text("Iteration Info", weight=BOLD).scale(0.5).move_to(panel_bg.get_top()).shift(DOWN*0.35)
        eq_update = MathTex(r"x_{k+1}=x_k-\frac{f'(x_k)}{f''(x_k)}").scale(0.6).next_to(panel_title, DOWN, buff=0.15)
        info_lines = VGroup(
            MathTex(r"x_k = \, ?").scale(0.6),
            MathTex(r"f'(x_k) = \, ?").scale(0.6),
            MathTex(r"f''(x_k) = \, ?").scale(0.6),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(eq_update, DOWN, buff=0.2).align_to(eq_update, LEFT)

        info_panel = VGroup(panel_bg, panel_title, eq_update, info_lines)
        info_panel.scale_to_fit_width(5.8)
        self.add(info_panel)

        # 初始点
        xk = self.x0
        x_dot = Dot(ax.coords_to_point(xk, self.f(xk)), color=RED, radius=0.08)  # 增大点的大小
        x_vline = ax.get_vertical_line(
            ax.coords_to_point(xk, self.f(xk)), 
            color=RED, 
            line_config={"stroke_opacity": 0.7, "stroke_width": 2}  # 加粗线条
        )
        x_label = MathTex("x_0").scale(0.7).next_to(x_vline, DOWN, buff=0.1)  # 增大标签
        self.play(FadeIn(x_dot), Create(x_vline), FadeIn(x_label), run_time=0.8)

        # 添加收敛进度指示器
        convergence_text = Text("Convergence Progress:", weight=BOLD).scale(0.5)
        convergence_marker = Line(LEFT, RIGHT, color=GREEN).scale(3).next_to(convergence_text, DOWN, buff=0.2)
        convergence_tip = Dot(convergence_marker.get_start(), color=GREEN).scale(0.8)
        convergence_group = VGroup(convergence_text, convergence_marker, convergence_tip).to_corner(UL, buff=0.5)
        self.add(convergence_group)

        quad_graph = None
        xk_marker = VGroup(x_dot, x_vline, x_label)
        x_history = [xk]  # 记录历史点用于展示收敛路径

        for k in range(self.max_iters):
            gk = self.df(xk)
            hk = self.d2f(xk)

            # 更新信息面板
            new_info = VGroup(
                MathTex(rf"x_k = \, {xk:.6f}").scale(0.6),  # 显示更多小数位
                MathTex(rf"f'(x_k) = \, {gk:.4e}").scale(0.6),
                MathTex(rf"f''(x_k) = \, {hk:.4e}").scale(0.6),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(info_lines, aligned_edge=LEFT)
            self.play(Transform(info_lines, new_info), run_time=0.6 * self.animation_speed)

            # 绘制二次近似曲线
            qk = taylor2_quad(self.f, self.df, self.d2f, xk)
            local_w = 2.0  # 扩大局部视图范围
            xa, xb = max(self.x_range[0], xk - local_w), min(self.x_range[1], xk + local_w)
            new_quad_graph = ax.plot(lambda t: qk(t), color=BLUE, x_range=(xa, xb), stroke_width=2)

            quad_label = MathTex(r"q_k(t)").scale(0.7).set_color(BLUE)
            quad_label.next_to(ax.coords_to_point(xb, qk(xb)), UR, buff=0.1)
            quad_label.add_background_rectangle(buff=0.04, opacity=0.6)

            if quad_graph is None:
                self.play(Create(new_quad_graph), FadeIn(quad_label), run_time=0.8 * self.animation_speed)
            else:
                self.play(ReplacementTransform(quad_graph, new_quad_graph), 
                          FadeTransform(quad_label_prev, quad_label), 
                          run_time=0.8 * self.animation_speed)
            quad_graph = new_quad_graph
            quad_label_prev = quad_label

            # 计算新点
            x_next, used_damping = safe_newton_step(self.df, self.d2f, xk)
            x_history.append(x_next)

            # 绘制新点
            x_next_dot = Dot(ax.coords_to_point(x_next, self.f(x_next)), color=GREEN, radius=0.08)
            x_next_vline = ax.get_vertical_line(
                ax.coords_to_point(x_next, self.f(x_next)), 
                color=GREEN, 
                line_config={"stroke_opacity": 0.7, "stroke_width": 2}
            )
            x_next_label = MathTex(fr"x_{{{k+1}}}").scale(0.7).set_color(GREEN).next_to(x_next_vline, DOWN, buff=0.1)

            # 显示迭代步长箭头
            step_arrow = Arrow(
                ax.coords_to_point(xk, self.y_range[0]+0.5),  # 箭头位置在x轴上方一点
                ax.coords_to_point(x_next, self.y_range[0]+0.5),
                color=YELLOW,
                stroke_width=2,
                tip_length=0.15
            )
            step_label = MathTex(fr"\Delta x = {x_next - xk:.4f}").scale(0.5).set_color(YELLOW)
            step_label.next_to(step_arrow, UP, buff=0.1)
            
            self.play(
                FadeIn(x_next_dot), Create(x_next_vline), FadeIn(x_next_label),
                Create(step_arrow), FadeIn(step_label),
                run_time=1.0 * self.animation_speed  # 延长步长展示时间
            )

            # 更新收敛进度指示器
            progress = min(1.0, k / (self.max_iters - 1))
            self.play(
                convergence_tip.animate.move_to(convergence_marker.point_from_proportion(progress)),
                run_time=0.3 * self.animation_speed
            )

            # 弱化历史点
            self.play(
                xk_marker.animate.set_opacity(0.2),
                run_time=0.3 * self.animation_speed
            )
            
            # 新点转为当前点
            xk_marker = VGroup(
                x_next_dot.copy().set_color(RED),
                x_next_vline.copy().set_color(RED),
                MathTex(fr"x_{{{k+1}}}").scale(0.7).next_to(x_next_vline, DOWN, buff=0.1).set_color(RED)
            )
            self.play(
                ReplacementTransform(x_next_dot, xk_marker[0]),
                ReplacementTransform(x_next_vline, xk_marker[1]),
                ReplacementTransform(x_next_label, xk_marker[2]),
                FadeOut(step_arrow),
                FadeOut(step_label),
                run_time=0.6 * self.animation_speed
            )

            # 更新当前点
            xk = x_next

        # 最终展示所有迭代点的收敛路径
        path_dots = VGroup(*[
            Dot(ax.coords_to_point(x, self.f(x)), color=RED, radius=0.05).set_opacity(0.6)
            for x in x_history
        ])
        path_lines = VGroup()
        for i in range(len(x_history)-1):
            path_lines.add(
                Line(
                    ax.coords_to_point(x_history[i], self.f(x_history[i])),
                    ax.coords_to_point(x_history[i+1], self.f(x_history[i+1])),
                    color=RED,
                    stroke_opacity=0.4,
                    stroke_width=1
                )
            )
        
        self.play(
            Create(path_lines),
            FadeIn(path_dots),
            run_time=1.5
        )
        
        # 显示收敛速度说明
        final_note = Text("Slow linear convergence near flat valley", color=ORANGE).scale(0.6)
        final_note.to_edge(DOWN, buff=0.5)
        final_note.add_background_rectangle(opacity=0.7)
        self.play(FadeIn(final_note), run_time=1.0)
        
        self.wait(2.0)  # 延长最终画面停留时间

# ---------- 平坦谷底案例：f(x) = x⁴ ----------
class NewtonFlatValley_X4(BaseNewton1D):
    title_text = "Newton (Flat Valley - Slow Convergence)"
    subtitle_text = r"$f(x)=x^4$  (only linear convergence near $x^*=0$)"
    x0 = 2.0
    x_range = (-3, 3, 1)
    y_range = (-0.5, 10, 1)
    max_iters = 10  # 增加迭代次数以明显展示缓慢收敛
    animation_speed = 1.5  # 进一步放慢动画
    
    def f(self, x): 
        return x**4  # x⁴函数在0附近有平坦谷底
    
    def df(self, x): 
        return 4*x**3  # 一阶导数
    
    def d2f(self, x): 
        return 12*x**2  # 二阶导数（在0附近非常小，导致收敛慢）

if __name__ == "__main__":
    # 可以直接运行此文件查看效果
    from manim import config
    config.media_dir = "./media"  # 设置媒体文件输出目录
    scene = NewtonFlatValley_X4()
    scene.render()
    