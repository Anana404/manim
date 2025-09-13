from manim import *
import numpy as np

# ---------- 1. 目标函数 ----------
def f(x):
    return (x - 1)**4 + 0.5*(x + 1)**2 + 1.0

def f1(x):
    return 4*(x - 1)**3 + (x + 1)

def f2(x):
    return 12*(x - 1)**2 + 1.0  # 恒为正

def taylor_quadratic_at(x0):
    fx0, f1x0, f2x0 = f(x0), f1(x0), f2(x0)
    return lambda x: fx0 + f1x0*(x - x0) + 0.5*f2x0*(x - x0)**2

class NewtonExtremumScene(Scene):
    def construct(self):
        # 缩放因子：整体缩小所有元素
        scale_factor = 0.85
        
        # 基础配置
        config.frame_width = 14 * scale_factor
        config.frame_height = 10 * scale_factor

        # ---------- 2. 坐标轴设置（缩小尺寸） ----------
        x_range = (-5, 5, 1)
        y_range = (0, 170, 20)
        ax = Axes(
            x_range=x_range, 
            y_range=y_range,
            x_length=10 * scale_factor,  # 按比例缩小
            y_length=7 * scale_factor,
            tips=False,
            axis_config={
                "label_direction": DOWN,
                "font_size": int(20 * scale_factor),  # 文字按比例缩小
            },
            y_axis_config={
                "label_direction": LEFT,
                "font_size": int(20 * scale_factor),
            }
        ).shift(LEFT*0.3)

        # 坐标轴标签（缩小并调整位置）
        x_label = ax.get_x_axis_label(MathTex("x").scale(scale_factor), 
                                     edge=RIGHT, direction=RIGHT*0.5)
        y_label = ax.get_y_axis_label(MathTex("f(x)").scale(scale_factor), 
                                     edge=UP, direction=UP*0.5)
        self.play(Create(ax), FadeIn(x_label, y_label))

        # ---------- 3. 原函数曲线（删除背景） ----------
        f_graph = ax.plot(
            f, 
            color=BLUE, 
            stroke_width=2.5 * scale_factor,
            x_range=(-5, 5, 0.01)
        )
        f_label = MathTex("f(x)= (x-1)^4 + \\tfrac{1}{2}(x+1)^2 + 1").scale(0.6 * scale_factor)
        # 直接设置文字位置，删除BackgroundRectangle
        f_label.to_edge(UP).to_edge(LEFT).shift(RIGHT*0.5 + DOWN*0.2)
        self.play(Create(f_graph), FadeIn(f_label))

        # ---------- 4. 右侧信息面板（删除白色背景） ----------
        panel_title = Text("牛顿法（求一维极值）", weight=BOLD).scale(0.45 * scale_factor)
        eq1 = MathTex("x_{k+1} = x_k - \\frac{f'(x_k)}{f''(x_k)}").scale(0.65 * scale_factor)
        eq2 = MathTex("q_k(x)=f(x_k)+f'(x_k)(x-x_k)+\\tfrac{1}{2}f''(x_k)(x-x_k)^2").scale(0.55 * scale_factor)
        note = Text("在 x_k 处用二次函数 q_k(x)\n近似 f(x)，并令 q_k'(x)=0 得到下一步 x_{k+1}", 
                   line_spacing=0.9).scale(0.35 * scale_factor)

        panel = VGroup(
            panel_title,
            eq1,
            eq2,
            note
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25 * scale_factor)

        # 删除RoundedRectangle背景，直接使用panel
        panel_grp = panel
        panel_grp.to_edge(RIGHT).shift(UP*1.2 * scale_factor)
        self.play(FadeIn(panel_grp))

        # 数值显示面板（删除背景）
        xk_tex = MathTex("x_k = \\,").scale(0.65 * scale_factor)
        xk_val = DecimalNumber(0.0, num_decimal_places=5).scale(0.65 * scale_factor)
        xkp1_tex = MathTex("x_{k+1} = \\,").scale(0.65 * scale_factor)
        xkp1_val = DecimalNumber(0.0, num_decimal_places=5).scale(0.65 * scale_factor)

        rows = VGroup(
            VGroup(xk_tex, xk_val).arrange(RIGHT, buff=0.15 * scale_factor),
            VGroup(xkp1_tex, xkp1_val).arrange(RIGHT, buff=0.15 * scale_factor),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2 * scale_factor)

        # 删除BackgroundRectangle，直接设置rows位置
        rows_grp = rows.to_edge(RIGHT).shift(DOWN*1.0 * scale_factor + LEFT*0.2)
        self.play(FadeIn(rows_grp))

        # ---------- 5. 迭代设置 ----------
        x0 = -3.0
        n_steps = 10
        dot_color = YELLOW
        quad_color = ORANGE
        brace_color = GREY_B

        def x_marker(x, color=dot_color, label_text=None):
            d = Dot(ax.c2p(x, 0), color=color, radius=0.055 * scale_factor)
            if label_text is None:
                label_text = f"x"
            lb = MathTex(label_text).scale(0.5 * scale_factor)
            lb.next_to(d, DOWN, buff=0.2 * scale_factor)
            return VGroup(d, lb)

        # ---------- 6. 初始点可视化 ----------
        xk = x0
        xk_dot = Dot(ax.c2p(xk, f(xk)), color=dot_color, radius=0.065 * scale_factor)
        xk_vline = ax.get_vertical_line(ax.c2p(xk, f(xk)), 
                                      line_func=Line, 
                                      color=brace_color, 
                                      stroke_width=2 * scale_factor)
        xk_marker_grp = x_marker(xk, color=dot_color, label_text="x_0")
        self.play(FadeIn(xk_dot), Create(xk_vline), FadeIn(xk_marker_grp))
        self.wait(0.5)

        xk_val.set_value(xk)
        xkp1_val.set_value(xk)

        # ---------- 7. 逐步迭代 ----------
        prev_quad_graph = None
        prev_arrow = None

        for k in range(n_steps):
            # 计算下一步
            f1x = f1(xk)
            f2x = f2(xk)
            if abs(f2x) < 1e-8:
                f2x = 1e-8
            x_next = xk - f1x / f2x

            # 当前二次近似曲线
            qk = taylor_quadratic_at(xk)
            quad_graph = ax.plot(
                qk, 
                color=quad_color, 
                z_index=1, 
                use_smoothing=False, 
                stroke_width=2.5 * scale_factor,
                x_range=(xk-2, xk+2, 0.01)
            )
            quad_label = MathTex(f"q_{k}(x)").scale(0.5 * scale_factor).set_color(quad_color)
            
            # 拟合曲线标签位置
            label_x = np.clip(xk + 0.8, x_range[0]+0.5, x_range[1]-0.5)
            label_y = np.clip(qk(label_x), y_range[0]+5, y_range[1]-5)
            quad_label.move_to(ax.c2p(label_x, label_y)).shift(UP*0.3 * scale_factor)

            # 二次近似注释（删除背景）
            arc = ArcBetweenPoints(
                ax.c2p(xk - 0.6, qk(xk - 0.6)),
                ax.c2p(xk + 0.6, qk(xk + 0.6)),
                angle=-1.2, color=brace_color
            ).set_opacity(0.6).scale(scale_factor)
            arc_note = Text("在 x_k 处的二次近似", 
                           font_size=int(20 * scale_factor)).set_color(GREY_D)
            # 删除BackgroundRectangle，直接使用arc_note
            arc_note_grp = arc_note
            arc_note_grp.next_to(quad_label, UP, buff=0.2 * scale_factor)

            # 下一个迭代点标记
            xkp1_marker_grp = x_marker(x_next, color=GREEN, label_text=fr"x_{{{k+1}}}")
            xkp1_dot = Dot(ax.c2p(x_next, f(x_next)), color=GREEN, radius=0.065 * scale_factor)
            xkp1_vline = ax.get_vertical_line(ax.c2p(x_next, f(x_next)), 
                                             line_func=Line, 
                                             color=brace_color, 
                                             stroke_width=2 * scale_factor)

            # 迭代箭头
            arrow = Arrow(
                ax.c2p(xk, 0), ax.c2p(x_next, 0),
                buff=0.0, 
                stroke_width=5 * scale_factor, 
                max_tip_length_to_length_ratio=0.12, 
                color=GREEN
            )

            # 清理上一轮元素
            anims = []
            if prev_quad_graph is not None:
                anims.append(FadeOut(prev_quad_graph))
            if prev_arrow is not None:
                anims.append(FadeOut(prev_arrow))
            if anims:
                self.play(*anims)

            # 展示当前二次拟合
            self.play(Create(quad_graph), FadeIn(quad_label), Create(arc), FadeIn(arc_note_grp))
            self.wait(0.6)

            # 展示迭代箭头
            self.play(Create(arrow))
            self.wait(0.3)

            # 高亮新点
            self.play(
                Create(xkp1_vline),
                FadeIn(xkp1_marker_grp),
                FadeIn(xkp1_dot),
                run_time=0.7
            )

            # 更新数值显示
            xk_val.set_value(xk)
            xkp1_val.set_value(x_next)

            self.wait(0.7)

            # 准备下一轮
            self.play(
                ReplacementTransform(xk_marker_grp.copy(), xkp1_marker_grp.copy()),
                run_time=0.4
            )

            # 清理当前标注
            self.play(FadeOut(quad_label), FadeOut(arc), FadeOut(arc_note_grp))

            # 更新引用
            prev_quad_graph = quad_graph
            prev_arrow = arrow

            # 移动当前点高亮
            self.play(
                Transform(xk_dot, xkp1_dot),
                Transform(xk_vline, xkp1_vline),
                Transform(xk_marker_grp, xkp1_marker_grp),
                run_time=0.8
            )
            xk = x_next

        # ---------- 8. 收尾总结（删除背景） ----------
        final_text = VGroup(
            Text("收敛到极小值点附近", weight=BOLD, 
                 font_size=int(22 * scale_factor)).scale(0.5),
            MathTex("f'(x^*)=0,\\quad f''(x^*)>0").scale(0.65 * scale_factor)
        ).arrange(DOWN, buff=0.25 * scale_factor)

        # 删除RoundedRectangle背景，直接使用final_text
        final_grp = final_text
        final_grp.next_to(panel_grp, DOWN, buff=0.5 * scale_factor)
        self.play(FadeIn(final_grp))
        self.wait(2)

        # 强调对比
        self.play(
            Indicate(f_graph, color=BLUE, scale_factor=1.02),
            Indicate(prev_quad_graph, color=quad_color, scale_factor=1.02),
            run_time=1.5
        )
        self.wait(1.5)
