from manim import *
import numpy as np


class ConjugateDirections(Scene):
    def construct(self):
        # 设置标题
        title = Text("共轭方向 (Conjugate Directions) 的几何意义",
                     font_size=36).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # 创建坐标系和等高线
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": BLUE},
        ).shift(DOWN*0.5)

        # 定义函数和矩阵A
        A = np.array([[4.0, 0.0], [0.0, 1.0]])

        def f(x, y):
            return 0.5 * (A[0, 0]*x*x + A[1, 1]*y*y)

        # 创建等高线（椭圆）
        contours = VGroup()
        levels = [1, 4, 9, 16]
        for level in levels:
            a = np.sqrt(2*level/A[0, 0])  # 椭圆x半轴
            b = np.sqrt(2*level/A[1, 1])  # 椭圆y半轴

            ellipse = Ellipse(
                width=axes.x_axis.unit_size * 2 * a,
                height=axes.y_axis.unit_size * 2 * b,
                color=WHITE,
                stroke_width=2,
                fill_opacity=0,
                stroke_opacity=0.7
            )
            ellipse.move_to(axes.c2p(0, 0))
            contours.add(ellipse)

        self.play(Create(axes), Create(contours))
        self.wait(1)

        # 起点和终点
        start_point = axes.c2p(4, 4)
        optimal_point = axes.c2p(0, 0)  # 最小值点

        start_dot = Dot(start_point, color=RED)
        optimal_dot = Dot(optimal_point, color=GREEN)

        self.play(Create(start_dot), Create(optimal_dot))

        # 标签
        start_label = Text("起点", font_size=20).next_to(start_dot, UR, buff=0.1)
        optimal_label = Text("最优解", font_size=20).next_to(
            optimal_dot, DL, buff=0.1)
        self.play(Write(start_label), Write(optimal_label))
        self.wait(1)

        # 展示最速下降法的低效路径
        review_text = Text("回顾: 最速下降法的'锯齿'路径", font_size=24,
                           color=YELLOW).to_edge(DOWN)
        self.play(Write(review_text))
        self.wait(1)

        # 快速模拟几步最速下降（简化版）
        x = np.array([4.0, 4.0])
        path_points = [start_point]

        for _ in range(4):
            gradient = A.dot(x)
            direction = -gradient
            alpha = (direction @ direction) / (direction @ A @ direction)
            x_new = x + alpha * direction
            new_point = axes.c2p(x_new[0], x_new[1])
            path_points.append(new_point)
            x = x_new

        # 绘制锯齿路径
        zigzag_path = VMobject()
        zigzag_path.set_points_as_corners(path_points)
        zigzag_path.set_stroke(color=RED, width=3)

        self.play(Create(zigzag_path), run_time=2)
        self.wait(2)

        # 清除回顾内容
        self.play(
            FadeOut(zigzag_path),
            FadeOut(review_text),
            run_time=1
        )

        # 现在展示共轭方向的魔力！
        magic_text = Text("但如果选择'正确'的方向呢？", font_size=28,
                          color=GREEN).to_edge(DOWN)
        self.play(Write(magic_text))
        self.wait(2)

        # 定义两个共轭方向
        d0 = np.array([1.0, 0.0])   # x方向
        d1 = np.array([0.0, 1.0])   # y方向 - 与d0关于A共轭：d0^T A d1 = 0

        # 可视化第一个方向d0
        d0_arrow = Arrow(
            start=start_point,
            end=axes.c2p(4 + 3*d0[0], 4 + 3*d0[1]),
            color=BLUE,
            buff=0,
            stroke_width=4
        )
        d0_label = MathTex("d_0", color=BLUE, font_size=28).next_to(
            d0_arrow, RIGHT, buff=0.1)

        self.play(Create(d0_arrow), Write(d0_label))
        self.wait(1)

        # 沿着d0方向找到最小值
        # 对于二次型，最优步长 α = - (∇f(x)^T d) / (d^T A d)
        gradient0 = A.dot(np.array([4.0, 4.0]))
        alpha0 = - (gradient0 @ d0) / (d0 @ A @ d0)
        intermediate_point = axes.c2p(4 + alpha0*d0[0], 4 + alpha0*d0[1])

        # 移动点到中间位置
        intermediate_dot = Dot(intermediate_point, color=ORANGE)
        self.play(Transform(start_dot, intermediate_dot), run_time=2)
        self.wait(1)

        # 可视化第二个方向d1（从中间点出发）
        d1_arrow = Arrow(
            start=intermediate_point,
            end=axes.c2p(
                intermediate_point[0] + 3*d1[0], intermediate_point[1] + 3*d1[1]),
            color=PURPLE,
            buff=0,
            stroke_width=4
        )
        d1_label = MathTex("d_1", color=PURPLE, font_size=28).next_to(
            d1_arrow, UP, buff=0.1)

        self.play(Create(d1_arrow), Write(d1_label))
        self.wait(1)

        # 沿着d1方向直接到达最优解！
        gradient1 = A.dot(np.array([4 + alpha0*d0[0], 4 + alpha0*d0[1]]))
        alpha1 = - (gradient1 @ d1) / (d1 @ A @ d1)
        final_point = axes.c2p(
            intermediate_point[0] + alpha1*d1[0], intermediate_point[1] + alpha1*d1[1])

        # 绘制完美路径
        perfect_path = VMobject()
        perfect_path.set_points_as_corners(
            [start_point, intermediate_point, final_point])
        perfect_path.set_stroke(color=GREEN, width=4)

        self.play(Create(perfect_path), run_time=1)
        self.play(Transform(start_dot, optimal_dot), run_time=1.5)  # 点移动到最优解
        self.wait(2)

        # 解释为什么这两个方向是特殊的
        explanation_text = Text("d₀ 和 d₁ 关于矩阵 A 共轭",
                                font_size=24, color=YELLOW).to_edge(DOWN).shift(UP * 0.8)

        math_explanation = MathTex(r"d_0^T A d_1 = 0",
                                   font_size=36, color=YELLOW).next_to(explanation_text, DOWN, buff=0.3)

        self.play(
            ReplacementTransform(magic_text, explanation_text),
            Write(math_explanation),
            run_time=1.5
        )
        self.wait(3)

        # 总结
        summary_text = Text("共轭方向确保了在每个方向上的一次性优化不会互相干扰",
                            font_size=22, color=BLUE).to_edge(DOWN).shift(UP*0.5)

        self.play(
            FadeOut(explanation_text),
            FadeOut(math_explanation),
            Write(summary_text),
            run_time=1.5
        )
        self.wait(3)

        # 清理场景，为下一个场景做准备
        self.play(
            FadeOut(VGroup(axes, contours, start_dot, optimal_dot, start_label, optimal_label,
                           d0_arrow, d0_label, d1_arrow, d1_label, perfect_path, summary_text, title)),
            run_time=2
        )


# 运行场景
if __name__ == "__main__":
    scene = ConjugateDirections()
    scene.render()
