from manim import *
import numpy as np
import math
import sympy as sp

class AdamOptimization(ThreeDScene):
    def construct(self):
        self.title = Text("Adam 优化算法演进", font_size=36, color=WHITE)
        self.play(Write(self.title))
        self.wait(1)
        self.play(self.title.animate.scale(0.8).to_edge(UP))
        
        # 创建损失函数地形
        self.setup_2d_landscape()
        
        # 展示Momentum算法
        self.show_momentum_solution()
        
        # 展示RMSProp算法
        self.show_rmsprop_solution()
        
        # 展示Adam算法
        self.show_adam_solution()
        
        # 算法比较
        self.show_algorithm_comparison()
        
        # 总结
        self.show_conclusion()

        # 应用
        self.showAdam3dVisualization()

    def setup_2d_landscape(self):
        axes = Axes(
            x_range=[1, 10.3, 1],
            y_range=[0, 25, 4],
            x_length=8,
            y_length=6,
            axis_config={"color": YELLOW},
            tips=False,
        ).to_edge(LEFT)
        
        axes_labels = axes.get_axis_labels(
            x_label=Text("参数空间", font_size=16, color=YELLOW),
            y_label=Text("损失值", font_size=16, color=YELLOW)
        )

        def loss_function(x):
            return 0.5 * (x-6)**2 + 2 * np.sin(x * 0.5 * math.pi) + 5
        
        loss_curve = axes.plot(loss_function, color=RED, x_range=[1.5, 9])
        
        self.play(Create(axes), Write(axes_labels), run_time=1.5)
        self.play(Create(loss_curve), run_time=2)
        self.wait(1)
        
        self.axes = axes
        self.loss_curve = loss_curve
        self.loss_function = loss_function

    def show_momentum_solution(self):

        # ────────────── 1. 场景清理与标题 ──────────────
        self.clear_text()

        momentum_title = Text("Momentum", font_size=20, color=GREEN).to_edge(UR).shift(DOWN*1.3).shift(LEFT*1.8)
        self.play(Write(momentum_title))

        momentum_benefits = VGroup(
            Text("• 引入动量项", font_size=20, color=GREEN),
            Text("• 加速收敛", font_size=20, color=GREEN),
            Text("• 所有参数共用同一学习率", font_size=20, color=RED),
            Text("  无法自适应", font_size=20, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(momentum_title, DOWN, buff=0.5)
        self.play(Write(momentum_benefits), run_time=2)

        # ────────────── 2. 真实梯度准备 ──────────────
        x_sym = sp.symbols('x')
        loss_sym = 0.5 * (x_sym - 6) ** 2 + 2 * sp.sin(x_sym * 0.5 * sp.pi) + 5
        grad_sym = sp.diff(loss_sym, x_sym)
        grad_fn = sp.lambdify(x_sym, grad_sym, 'numpy')

        # ────────────── 3. 小球与路径初始化 ──────────────
        start_x = 1.5
        momentum_dot = Dot(
            self.axes.coords_to_point(start_x, self.loss_function(start_x)),
            color=GREEN, radius=0.15
        )
        self.add(momentum_dot)

        momentum_label = Text("Momentum", font_size=20, color=GREEN)
        momentum_label.next_to(momentum_dot, LEFT + DOWN, buff=0.3)

        momentum_label.add_updater(
            lambda m: m.next_to(momentum_dot, LEFT + DOWN, buff=0.3)
        )

        self.add(momentum_label)

        momentum_path = VMobject(color=GREEN, stroke_width=6)

        # ────────────── 4. 流畅版 Momentum 迭代 ──────────────
        current_x = start_x
        velocity  = 0.0
        lr, gamma = 0.005, 0.9
        steps = 50
        path_x = [current_x]

        for _ in range(steps):
            grad = grad_fn(current_x)
            velocity = gamma * velocity + lr * grad
            current_x -= velocity
            path_x.append(current_x)

        smooth_points = [
            self.axes.coords_to_point(x, self.loss_function(x)) for x in path_x
        ]
        momentum_path_smooth = VMobject(color=GREEN, stroke_width=6)
        momentum_path_smooth.set_points_smoothly(smooth_points)

        self.play(
            MoveAlongPath(momentum_dot, momentum_path_smooth),
            Create(momentum_path_smooth),
            run_time=4,
            rate_func=smooth
        )

        # ────────────── 5. 保存引用 ──────────────
        self.momentum_title = momentum_title
        self.momentum_benefits = momentum_benefits
        self.momentum_dot = momentum_dot
        self.momentum_label = momentum_label
        self.momentum_path = momentum_path

        self.clear_text(self.momentum_title, self.momentum_benefits)


    def show_rmsprop_solution(self):
        # ────────────── 1. 场景清理与标题 ──────────────
        # self.clear_text()

        rmsprop_title = Text("RMSProp", font_size=20, color=PURPLE).to_edge(UR).shift(DOWN*1.3).shift(LEFT*1.8)
        self.play(Write(rmsprop_title))

        rmsprop_benefits = VGroup(
            Text("• 自适应学习率", font_size=20, color=PURPLE),
            Text("• 对不同参数方向自适应", font_size=20, color=PURPLE),
            Text("• 缺乏动量，可能在鞍点处停滞，", font_size=20, color=PURPLE),
            Text("  优化路径可能不平滑", font_size=20, color=PURPLE)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(rmsprop_title, DOWN, buff=0.5)
        self.play(Write(rmsprop_benefits), run_time=2)

        # ────────────── 2. 真实梯度准备 ──────────────
        x_sym = sp.symbols('x')
        loss_sym = 0.5 * (x_sym - 6)**2 + 2 * sp.sin(x_sym * 0.5 * sp.pi) + 5
        grad_sym = sp.diff(loss_sym, x_sym)               # 真实梯度 g_t
        grad_fn = sp.lambdify(x_sym, grad_sym, 'numpy')   # 可调用函数

        # ────────────── 3. 小球与路径初始化 ──────────────
        start_x = 1.5
        rmsprop_dot = Dot(
            self.axes.coords_to_point(start_x, self.loss_function(start_x)),
            color=PURPLE, radius=0.15
        )
        self.add(rmsprop_dot)

        rmsprop_label = Text("RMSProp", font_size=20, color=PURPLE)
        rmsprop_label.next_to(rmsprop_dot, RIGHT + DOWN, buff=0.3)

        rmsprop_label.add_updater(
            lambda m: m.next_to(rmsprop_dot, RIGHT + DOWN, buff=0.3)
        )

        self.add(rmsprop_label)

        rmsprop_path = VMobject(color=PURPLE, stroke_width=6)
        rmsprop_points = [rmsprop_dot.get_center()]

        # ────────────── 4. 流畅版 RMSProp 迭代 ──────────────
        current_x   = start_x
        avg_sq_grad = 0.0
        lr, gamma, eps = 0.5, 0.9, 1e-8
        steps = 30
        path_x = [current_x]

        for t in range(1, steps + 1):
            grad = grad_fn(current_x)
            avg_sq_grad = gamma * avg_sq_grad + (1 - gamma) * grad**2
            step = lr * grad / (np.sqrt(avg_sq_grad) + eps)
            current_x -= step
            path_x.append(current_x)

        smooth_points = [
            self.axes.coords_to_point(x, self.loss_function(x)) for x in path_x
        ]
        rmsprop_path_smooth = VMobject(color=PURPLE, stroke_width=6)
        rmsprop_path_smooth.set_points_smoothly(smooth_points)

        self.play(
            MoveAlongPath(rmsprop_dot, rmsprop_path_smooth),
            Create(rmsprop_path_smooth),
            run_time=4, 
            rate_func=smooth
        )

        # ────────────── 5. 保存引用 ──────────────
        self.rmsprop_title      = rmsprop_title
        self.rmsprop_benefits   = rmsprop_benefits
        self.rmsprop_dot        = rmsprop_dot
        self.rmsprop_label      = rmsprop_label
        self.rmsprop_path       = rmsprop_path

        self.clear_text(self.rmsprop_title, self.rmsprop_benefits)


    def show_adam_solution(self):
        # ────────────── 1. 场景清理与标题 ──────────────
        # self.clear_text()

        adam_title = Text("Adam", font_size=20, color=ORANGE).to_edge(UR).shift(DOWN*0.7).shift(LEFT*1.8)
        self.play(Write(adam_title))

        adam_benefits = VGroup(
            Text("• 结合Momentum和RMSProp", font_size=20, color=ORANGE),
            Text("• 动量加速收敛", font_size=20, color=ORANGE),
            Text("• 自适应学习率", font_size=20, color=ORANGE),
            Text("• 偏差校正提高稳定性", font_size=20, color=ORANGE)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(adam_title, DOWN, buff=0.5)
        self.play(Write(adam_benefits), run_time=2)

        font_size = 24
        line1 = MathTex(r"m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t", color=GREEN, font_size=font_size)
        line2 = MathTex(r"v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2", color=PURPLE, font_size=font_size)
        line3 = MathTex(r"\hat{m}_t = \frac{m_t}{1-\beta_1^t}", color=BLUE, font_size=font_size)
        line4 = MathTex(r"\hat{v}_t = \frac{v_t}{1-\beta_2^t}", color=BLUE, font_size=font_size)
        line5 = MathTex(r"\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t}+\varepsilon} \hat{m}_t", color=WHITE, font_size=font_size)

        adam_formula = VGroup(line1, line2, line3, line4, line5)\
                   .arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(adam_benefits, DOWN, buff=0.5)

        self.play(Write(adam_formula))
        self.wait(1)

        # ────────────── 2. 真实梯度准备 ──────────────
        x_sym = sp.symbols('x')
        loss_sym = 0.5 * (x_sym - 6)**2 + 2 * sp.sin(x_sym * 0.5 * sp.pi) + 5
        grad_sym = sp.diff(loss_sym, x_sym)
        grad_fn = sp.lambdify(x_sym, grad_sym, 'numpy')

        # ────────────── 3. 小球与路径初始化 ──────────────
        start_x = 1.5
        adam_dot = Dot(
            self.axes.coords_to_point(start_x, self.loss_function(start_x)),
            color=ORANGE, radius=0.15
        )
        self.add(adam_dot)

        adam_label = Text("Adam", font_size=20, color=ORANGE)
        adam_label.next_to(adam_dot, DOWN, buff=0.3)

        adam_label.add_updater(
            lambda m: m.next_to(adam_dot, DOWN, buff=0.3)
        )

        self.add(adam_label)

        adam_path = VMobject(color=ORANGE, stroke_width=6)
        adam_points = [adam_dot.get_center()]

        # ────────────── 4. 真实 Adam 迭代 ──────────────
        current_x = start_x
        m, v = 0.0, 0.0
        beta1, beta2 = 0.9, 0.999
        lr, eps = 0.5, 1e-8
        steps = 500
        path_x = [current_x]

        for t in range(1, steps + 1):
            g = grad_fn(current_x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g ** 2
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            current_x -= lr * m_hat / (np.sqrt(v_hat) + eps)
            path_x.append(current_x)

        smooth_points = [
            self.axes.coords_to_point(x, self.loss_function(x)) for x in path_x
        ]
        adam_path_smooth = VMobject(color=ORANGE, stroke_width=6)
        adam_path_smooth.set_points_smoothly(smooth_points)  # 三次样条光滑

        self.play(
            MoveAlongPath(adam_dot, adam_path_smooth),
            Create(adam_path_smooth),
            run_time=4,
            rate_func=smooth
        )

        # ────────────── 5. 保存引用 ──────────────
        self.adam_title   = adam_title
        self.adam_benefits = adam_benefits
        self.adam_formula  = adam_formula
        self.adam_dot      = adam_dot
        self.adam_label    = adam_label
        self.adam_path     = adam_path

        self.clear_text(self.adam_title, self.adam_benefits, self.adam_formula)

    def show_algorithm_comparison(self):
        comparison_title = Text("算法性能比较", font_size=20, color=YELLOW).to_edge(UR).shift(DOWN*1.0).shift(LEFT*1.8)
        
        self.play(
            self.momentum_path.animate.set_stroke(width=3),
            self.rmsprop_path.animate.set_stroke(width=3),
            self.adam_path.animate.set_stroke(width=3),
            run_time=2
        )
        
        legend = VGroup(
            Text("Momentum", color=GREEN, font_size=15),
            Text("RMSProp", color=PURPLE, font_size=15),
            Text("Adam", color=ORANGE, font_size=15)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).to_edge(UL).shift(RIGHT*2).shift(DOWN*0.5)
        
        legend_dots = VGroup(
            Dot(color=GREEN, radius=0.1).next_to(legend[0], LEFT, buff=0.3),
            Dot(color=PURPLE, radius=0.1).next_to(legend[1], LEFT, buff=0.3),
            Dot(color=ORANGE, radius=0.1).next_to(legend[2], LEFT, buff=0.3)
        )
        
        self.play(Write(legend), Write(legend_dots))
        self.wait(2)
        
        comparison_results = VGroup(
            Text("Adam结合了Momentum和RMSProp的优点:", font_size=20, color=ORANGE),
            Text("• 动量加速收敛", font_size=20, color=GREEN),
            Text("• 自适应学习率防止震荡", font_size=20, color=PURPLE),
            Text("• 偏差校正提高稳定性", font_size=20, color=ORANGE),
            Text("• 适用于大多数深度学习问题", font_size=20, color=ORANGE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(comparison_title, DOWN, buff=0.5)
        
        self.play(Write(comparison_results), run_time=3)
        self.wait(3)

        # ────────────── 保存引用 ──────────────
        self.comparison_title    = comparison_title
        self.comparison_results  = comparison_results
        self.legend_dots         = legend_dots
        self.legend              = legend

        self.clear_text(self.comparison_results)

    def show_conclusion(self):
        conclusion_title = Text("总结", font_size=20, color=BLUE).to_edge(UR).shift(DOWN*1.3).shift(LEFT*1.8)
        
        conclusion_points = VGroup(
            Text("Adam优化算法的核心优势:", font_size=20, color=ORANGE),
            Text("1. 结合了Momentum的动量加速", font_size=20, color=GREEN),
            Text("2. 结合了RMSProp的自适应学习率", font_size=20, color=PURPLE),
            Text("3. 引入偏差校正提高训练稳定性", font_size=20, color=ORANGE),
            Text("4. 对超参数选择相对鲁棒", font_size=20, color=ORANGE),
            Text("5. 适用于大多数深度学习任务", font_size=20, color=ORANGE)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(conclusion_title, DOWN, buff=0.5)
        
        self.play(Write(conclusion_points), run_time=3)
        self.wait(3)
        
        self.clear()
        
        final_message = Text(
            "Adam已成为深度学习中最受欢迎的优化算法之一",
            font_size=20,
            color=GREEN
        )
        
        self.play(Write(final_message))
        self.wait(3)
        self.conclusion_points = conclusion_points
        self.conclusion_title = conclusion_title
        self.final_message = final_message
        self.clear_text(self.final_message)

    def showAdam3dVisualization(self):
        adam_apply_title = Text("Adam的简单应用", font_size=30, color=WHITE)
        self.adam_apply_title = adam_apply_title
        self.play(Write(adam_apply_title))
        adam_apply_formula = MathTex(
            r"f(x, y) = (x - 2)^2 + 2(y - 3)^2",
            font_size=28
        )
        adam_apply_formula.next_to(adam_apply_title, DOWN, buff=0.5)
        
        self.play(Write(adam_apply_formula))
        self.wait(1)
        self.play(adam_apply_formula.animate.scale(0.8).to_edge(LEFT+UP))
        self.add_fixed_in_frame_mobjects(adam_apply_formula)
        self.wait(1)
        self.clear_text(self.adam_apply_title)

        # ======= 超参数、初始化保持不变 =======
        lr = 0.1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        num_iterations = 70

        params = np.array([0.0, 0.0])
        m, v = np.zeros(2), np.zeros(2)
        loss_list, params_list, grad_norm_list = [], [], []

        for t in range(1, num_iterations + 1):
            grad = self.grad_f(params[0], params[1])
            loss_list.append(self.f(params[0], params[1]))
            params_list.append(params.copy())
            grad_norm_list.append(np.linalg.norm(grad))

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        params_list = np.array(params_list)

        # ======= 场景、坐标系、曲面等保持不变 =======
        self.set_camera_orientation(phi=45 * DEGREES, theta=45 * DEGREES, zoom=0.8)
        self.camera.frame_center += np.array([0, 0, 3])
        axes = ThreeDAxes(
            x_range=[-1, 5, 1],
            y_range=[-1, 7, 1],
            z_range=[0, 50, 10],
            x_length=8, y_length=8, z_length=6
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="f(x,y)")

        surface = Surface(
            lambda u, v: axes.c2p(u, v, self.f(u, v)),
            u_range=[-1, 5], v_range=[-1, 7],
            resolution=(30, 30),
            fill_opacity=0.6,
            stroke_width=0.5,
            fill_color=BLUE
        )

        optimal_point = Dot3D(axes.c2p(2, 3, 0), color=RED, radius=0.1)
        optimal_label = Text("Optimal (2,3)", font_size=20).next_to(optimal_point, UR, buff=0.1)
        start_point = Dot3D(axes.c2p(0, 0, self.f(0, 0)), color=YELLOW, radius=0.1)

        self.add(axes, axes_labels, surface, optimal_point, optimal_label, start_point)
        self.wait(1)

        # 在初始化之后、动画之前一次性算好密集插值
        dense_n = 500
        dense_t = np.linspace(0, len(params_list) - 1, dense_n)
        dense_pos = np.array([np.interp(dense_t, np.arange(len(params_list)), params_list[:, d])
                            for d in (0, 1)])
        dense_loss = np.interp(dense_t, np.arange(len(loss_list)), loss_list)
        dense_grad = np.interp(dense_t, np.arange(len(grad_norm_list)), grad_norm_list)

        # --------------- 一次性创建对象 ---------------
        tracker = ValueTracker(0)
        moving_dot = always_redraw(
            lambda: Dot3D(axes.c2p(dense_pos[0, int(tracker.get_value())],
                                dense_pos[1, int(tracker.get_value())],
                                self.f(dense_pos[0, int(tracker.get_value())],
                                        dense_pos[1, int(tracker.get_value())])),
                        color=YELLOW, radius=0.08)
        )
        self.add(moving_dot)

        # ---------- 4.2 预生成光滑模板 ----------
        template_path = VMobject()
        template_path.set_points_smoothly([
            axes.c2p(dense_pos[0, i], dense_pos[1, i],
                    self.f(dense_pos[0, i], dense_pos[1, i]))
            for i in range(dense_n)
        ])

        # ---------- 4.3 实际显示的路径（初始为空） ----------
        vis_path = VMobject(stroke_color=YELLOW, stroke_width=4)
        self.add(vis_path)

        # ---- 4.3 信息条 ----
        info_text = always_redraw(
            lambda: Text(
                f"Iteration: {int(dense_t[int(tracker.get_value())]) + 1}\n"
                f"Position: ({dense_pos[0, int(tracker.get_value())]:.2f}, "
                f"{dense_pos[1, int(tracker.get_value())]:.2f})\n"
                f"Loss: {dense_loss[int(tracker.get_value())]:.2f}\n"
                f"Gradient: {dense_grad[int(tracker.get_value())]:.2f}",
                font_size=20
            ).to_corner(UL).next_to(adam_apply_formula, DOWN, buff=0.5)
        )
        self.add_fixed_in_frame_mobjects(info_text)
        self.play(tracker.animate.set_value(dense_n - 1),
                run_time=4, rate_func=linear)
        self.wait(0.5)

        # ---------- 4.4 每帧把模板的前 alpha 段拷给显示路径 ----------
        def update_vis_path(m):
            alpha = tracker.get_value() / (dense_n - 1)
            m.become(template_path)
            m.set_color(YELLOW)
            m.pointwise_become_partial(template_path, 0, alpha)
        vis_path.add_updater(update_vis_path)


        # ======= 2D 损失曲线场景保持不变 =======
        self.move_camera(phi=0, theta=-90 * DEGREES, frame_center=ORIGIN, run_time=2)

        loss_axes = Axes(
            x_range=[0, num_iterations, 10],
            y_range=[0, max(loss_list), max(loss_list) / 8],
            x_length=6, y_length=4,
            axis_config={"stroke_width": 4, "color": GREEN}
        ).to_edge(DOWN)
        loss_axes_labels = loss_axes.get_axis_labels(
            x_label=Tex("\\textbf{Iteration}"),
            y_label=Tex("\\textbf{Loss}")
        )
        loss_axes_labels.set_color(GREEN).scale(0.8) 

        loss_graph = VMobject(stroke_color=RED, stroke_width=6)
        loss_graph.set_points_as_corners([
            loss_axes.coords_to_point(i, loss) for i, loss in enumerate(loss_list)
        ])

        self.play(Create(loss_axes), Write(loss_axes_labels))
        self.play(Create(loss_graph), run_time=2)
        self.wait(3)

    def f(self, x, y):
        return (x - 2)**2 + 2 * (y - 3)**2

    def grad_f(self, x, y):
        df_dx = 2 * (x - 2)
        df_dy = 4 * (y - 3)
        return np.array([df_dx, df_dy])

    def clear_text(self, *text_args):
        explicit = []
        for arg in text_args:
            if isinstance(arg, Mobject):
                explicit.append(arg)
            elif isinstance(arg, (list, tuple, VGroup)):
                explicit.extend([m for m in arg if isinstance(m, Mobject)])

        if not explicit:
            for attr in dir(self):
                if attr.endswith(('_title', '_benefits', '_formula', '_problems')):
                    try:
                        obj = getattr(self, attr)
                        if isinstance(obj, Mobject):
                            explicit.append(obj)
                        elif isinstance(obj, VGroup):
                            explicit.extend(obj)
                    except AttributeError:
                        continue
        if explicit:
            self.play(*[FadeOut(mob) for mob in explicit], run_time=1)
        else:
            self.wait(0.1)

if __name__ == "__main__":
    scene = AdamOptimization()
    scene.render()