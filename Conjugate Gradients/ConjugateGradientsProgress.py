from manim import *
import numpy as np


class ConjugateGradientProcess(Scene):
    def construct(self):

        # 创建坐标系和等高线
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": BLUE},
        ).shift(LEFT * 3)

        A = np.array([[4.0, 0.0], [0.0, 1.0]])
        b = np.array([0.0, 0.0])

        # 创建等高线
        contours = VGroup()
        levels = [1, 4, 9, 16]
        for level in levels:
            a = np.sqrt(2*level/A[0, 0])
            b_val = np.sqrt(2*level/A[1, 1])
            ellipse = Ellipse(
                width=axes.x_axis.unit_size * 2 * a,
                height=axes.y_axis.unit_size * 2 * b_val,
                color=WHITE,
                stroke_width=2,
                fill_opacity=0,
                stroke_opacity=0.7
            )
            ellipse.move_to(axes.c2p(0, 0))
            contours.add(ellipse)

        self.play(Create(axes), Create(contours))
        self.wait(1)

        # 创建公式面板（右侧）
        formula_box = Rectangle(
            width=4, height=6, color=WHITE, stroke_width=2, fill_color=BLACK, fill_opacity=0.8
        ).shift(RIGHT * 3)

        formula_title = Text("算法步骤", font_size=24, color=WHITE).next_to(
            formula_box, UP, buff=0.4)
        self.play(Create(formula_box), Write(formula_title))
        self.wait(1)

        # 初始化变量
        x = np.array([4.0, 4.0])  # 初始点
        r = A.dot(x) - b  # 初始残差
        d = -r.copy()     # 初始搜索方向

        current_point = axes.c2p(x[0], x[1])
        dot = Dot(current_point, color=RED)
        path = TracedPath(dot.get_center, stroke_color=RED, stroke_width=3)

        self.play(Create(dot))
        self.add(path)

        # 显示当前迭代信息
        iter_text = Text("迭代: 0", font_size=20, color=WHITE).next_to(
            formula_box, UP, buff=0.5).shift(DOWN * 0.5)
        self.play(Write(iter_text))

        # 算法步骤展示
        steps = VGroup()
        step_y_position = 2.0

        for iteration in range(2):  # 演示2次完整迭代

            # 步骤1: 计算步长 α
            step1_text = MathTex(
                r"\alpha = \frac{r^T r}{d^T A d}}",
                font_size=28,
                color=YELLOW
            ).move_to(formula_box.get_center() + UP * (step_y_position - 1.5))

            # 计算具体数值：α = (残差的内积) / (方向与A作用后的内积)
            # 这个步长确保在当前方向d上下降到最低点
            alpha = (r @ r) / (d @ A @ d)
            alpha_value = MathTex(
                f"\\alpha = {alpha:.3f}",
                font_size=24,
                color=GREEN
            ).next_to(step1_text, DOWN, buff=0.2)

            # 添加说明文本：解释这个步骤的意义
            step1_explanation = Text(
                "计算最优步长：使函数在当前方向下降最多",
                font_size=16,
                color=WHITE
            ).next_to(alpha_value, DOWN, buff=0.1)

            self.play(Write(step1_text), Write(
                alpha_value), Write(step1_explanation))
            steps.add(step1_text)
            self.wait(1.5)  # 稍微延长等待时间，让观众阅读说明

            # 只保留公式，移除说明文字
            self.play(
                FadeOut(step1_explanation, alpha_value),
                step1_text.animate.move_to(
                    formula_box.get_center() + UP * (step_y_position + 0.5)),
                run_time=0.5
            )

            # 步骤2: 更新解 x
            step2_text = MathTex(
                r"x_{new} = x + \alpha d",
                font_size=28,
                color=YELLOW
            ).move_to(formula_box.get_center() + UP * (step_y_position - 1.5))

            x_new = x + alpha * d
            x_value = MathTex(
                f"x_{{{iteration+1}}} = ({x_new[0]:.2f}, {x_new[1]:.2f})",
                font_size=24,
                color=GREEN
            ).next_to(step2_text, DOWN, buff=0.2)

            # 添加说明文本
            step2_explanation = Text(
                "沿搜索方向移动：更新当前解的位置",
                font_size=16,
                color=WHITE
            ).next_to(x_value, DOWN, buff=0.1)

            self.play(Write(step2_text), Write(
                x_value), Write(step2_explanation))
            steps.add(step2_text)
            self.wait(2)

            # 动画：移动点到新位置
            new_point = axes.c2p(x_new[0], x_new[1])
            self.play(dot.animate.move_to(new_point), run_time=2)
            self.wait(1)

            # 同样处理步骤2
            self.play(
                FadeOut(step2_explanation, x_value),
                step2_text.animate.move_to(
                    formula_box.get_center() + UP * (step_y_position)),
                run_time=0.5
            )

            # 步骤3: 更新残差 r
            step3_text = MathTex(
                r"r_{new} = r + \alpha A d",
                font_size=28,
                color=YELLOW
            ).move_to(formula_box.get_center() + UP * (step_y_position - 1.5))

            r_new = r + alpha * A.dot(d)
            r_value = MathTex(
                f"r_{{{iteration+1}}} = ({r_new[0]:.2f}, {r_new[1]:.2f})",
                font_size=24,
                color=GREEN
            ).next_to(step3_text, DOWN, buff=0.2)

            # 添加说明文本
            step3_explanation = Text(
                "更新残差：衡量当前解与最优解的差距",
                font_size=16,
                color=WHITE
            ).next_to(r_value, DOWN, buff=0.1)

            self.play(Write(step3_text), Write(
                r_value), Write(step3_explanation))
            steps.add(step3_text)
            self.wait(2)

            # 同样处理步骤2
            self.play(
                FadeOut(step3_explanation, r_value),
                step3_text.animate.move_to(
                    formula_box.get_center() + UP * (step_y_position - 0.5)),
                run_time=0.5
            )
            # 步骤4: 计算β和新的搜索方向（仅第一次迭代后显示）
            if iteration == 0:
                # 步骤4: 计算β（共轭参数）
                step4_text = MathTex(
                    r"\beta = \frac{r_{new}^T r_{new}}{r^T r}",
                    font_size=28,
                    color=YELLOW
                ).move_to(formula_box.get_center() + UP * (step_y_position - 1.5))

                beta = (r_new @ r_new) / (r @ r)
                beta_value = MathTex(
                    f"\\beta = {beta:.3f}",
                    font_size=24,
                    color=GREEN
                ).next_to(step4_text, DOWN, buff=0.2)

                # 添加说明文本
                step4_explanation = Text(
                    "计算共轭参数：确保新方向与旧方向共轭",
                    font_size=16,
                    color=WHITE
                ).next_to(beta_value, DOWN, buff=0.1)

                self.play(Write(step4_text), Write(
                    beta_value), Write(step4_explanation))
                self.wait(2)

                # 同样处理步骤2
                self.play(
                    FadeOut(step4_explanation, beta_value),
                    step4_text.animate.move_to(
                        formula_box.get_center() + UP * (step_y_position - 1.0)),
                    run_time=0.5
                )

                # 步骤5: 更新搜索方向
                step5_text = MathTex(
                    r"d_{new} = -r_{new} + \beta d",
                    font_size=28,
                    color=YELLOW
                ).move_to(formula_box.get_center() + UP * (step_y_position - 1.5))

                d_new = -r_new + beta * d
                d_value = MathTex(
                    f"d_{{{iteration+1}}} = ({d_new[0]:.2f}, {d_new[1]:.2f})",
                    font_size=24,
                    color=GREEN
                ).next_to(step5_text, DOWN, buff=0.2)

                # 添加说明文本
                step5_explanation = Text(
                    "生成新搜索方向：当前梯度与旧方向的组合",
                    font_size=16,
                    color=WHITE
                ).next_to(d_value, DOWN, buff=0.1)

                self.play(Write(step5_text), Write(
                    d_value), Write(step5_explanation))
                steps.add(step4_text, step5_text)
                self.wait(3)  # 给观众更多时间理解这个关键步骤

                # 同样处理步骤2
                self.play(
                    FadeOut(step5_explanation, d_value),
                    step5_text.animate.move_to(
                        formula_box.get_center() + UP * (step_y_position - 1.5)),
                    run_time=0.5
                )
            # 更新迭代计数器
            new_iter_text = Text(
                f"迭代: {iteration+1}", font_size=20, color=WHITE).move_to(iter_text)
            self.play(Transform(iter_text, new_iter_text))

            # 准备下一次迭代
            x = x_new
            r = r_new
            if iteration == 0:
                d = d_new

            # 清除步骤显示，为下一次迭代准备空间
            if iteration == 0:
                self.play(FadeOut(steps))
                steps = VGroup()
                self.wait(1)

        # 总结：共轭梯度法的优势
        summary_text = Text(
            "共轭梯度法自动生成共轭方向\n"
            "无需预先知道矩阵A的特征结构\n"
            "在n步内收敛到精确解（理论值）",
            font_size=20,
            color=WHITE
        ).to_edge(DOWN).shift(DOWN * 0.5)

        self.play(Write(summary_text))
        self.wait(3)

        # 最终清理
        self.play(*[FadeOut(obj) for obj in self.mobjects])


if __name__ == "__main__":
    scene = ConjugateGradientProcess()
    scene.render()
