from manim import *
import numpy as np

class OptimizedGradientDescentVisualization(Scene):
    def construct(self):
        # 第一部分：介绍
        title = Text("梯度下降法可视化", font_size=48).to_edge(UP)
        #subtitle = Text("理解优化算法的基础", font_size=36).next_to(title, DOWN)
        
        self.play(Write(title))
        self.wait(1)
        # self.play(Write(subtitle))
        # self.wait(2)
        
        # 添加介绍字幕
        intro_subtitles = [
            "大家好，今天我们将通过可视化来理解优化算法中的基础但非常重要的方法——梯度下降法。",
            "在实际应用中，我们经常需要找到函数的最小值点",
            "比如机器学习中的损失函数最小化、工程中的最优化设计。",
            "梯度下降法就是一种迭代优化算法，用于寻找可微函数的局部最小值。",
            "它的核心思想非常直观：如果我们想要下山，最有效的方法就是沿着最陡峭的方向向下走。",
            "在数学上，这个'最陡峭的方向'就是函数的负梯度方向。"
        ]
        
        for subtitle_text in intro_subtitles:
            subtitle_obj = Text(subtitle_text, font_size=20, color=YELLOW).to_edge(DOWN)
            self.play(Write(subtitle_obj))
            self.wait(1)
            self.play(FadeOut(subtitle_obj))
        
        self.play(FadeOut(title))
        
        # 第二部分：数学原理
        math_title = Text("梯度下降法数学原理", font_size=36).to_edge(UP)
        gradient_formula = MathTex("\\nabla f(x) = \\left[\\frac{\\partial f}{\\partial x_1}, \\frac{\\partial f}{\\partial x_2}, \\ldots, \\frac{\\partial f}{\\partial x_n}\\right]^T").scale(0.8)
        update_formula = MathTex("x_{k+1} = x_k - \\alpha \\nabla f(x_k)").scale(0.8)
        update_formula.next_to(gradient_formula, DOWN, buff=0.5)
        
        self.play(Write(math_title))
        self.wait(1)
        
        # 添加数学原理字幕
        math_subtitles = [
            "让我们正式定义梯度下降法。对于一个多元函数f(x)，在点x处的梯度∇f(x)指向函数增长最快的方向。",
            "因此，负梯度-∇f(x)指向函数下降最快的方向。",
            "梯度下降的更新公式是：x_{k+1} = x_k - α * ∇f(x_k)，其中α是学习率，控制每一步的步长大小。",
            "这个简单的公式是许多现代优化算法的基础。",
            "今天我们将以简单的二次函数f(x) = x² + 2x + 1为例，可视化梯度下降的过程。"
        ]
        
        for i, subtitle_text in enumerate(math_subtitles):
            subtitle_obj = Text(subtitle_text, font_size=20, color=YELLOW).to_edge(DOWN)
            
            if i == 0:
                self.play(Write(gradient_formula))
            elif i == 2:
                self.play(Write(update_formula))
                
            self.play(Write(subtitle_obj))
            self.wait(1)    
            self.play(FadeOut(subtitle_obj))
        
        self.play(FadeOut(math_title), FadeOut(gradient_formula), FadeOut(update_formula))
        
        # 第三部分：可视化设置
        # 创建坐标轴
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 10, 2],
            x_length=8,
            y_length=5,
            axis_config={"color": BLUE},
            x_axis_config={
                "numbers_to_include": np.arange(-3, 4, 1),
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, 11, 2),
            },
            tips=False,
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="f(x)")

        # 定义函数 f(x) = x^2 + 2x + 1
        def func(x):
            return x**2 + 2*x + 1

        # 创建函数曲线
        graph = axes.plot(func, color=WHITE, x_range=[-3, 3])

        # 添加函数标签
        func_label = MathTex("f(x) = x^2 + 2x + 1").to_corner(UL)
        
        # 显示最小值点
        min_x = -1
        min_point = Dot(axes.coords_to_point(min_x, func(min_x)), color=GREEN)
        min_label = Text("最小值点", font_size=20).next_to(min_point, RIGHT)
        
        # 添加可视化设置字幕
        setup_subtitles = [
            "首先，我们设置坐标系和函数曲线。",
            "这个二次函数有一个明显的最小值点。",
            "我们可以通过求导找到它的精确解。"
        ]
        
        for i, subtitle_text in enumerate(setup_subtitles):
            subtitle_obj = Text(subtitle_text, font_size=20, color=YELLOW).to_edge(DOWN)
    
            if i == 0:
                self.play(Create(axes), Write(axes_labels))
            elif i == 1:
                self.play(Create(graph), Write(func_label))
            elif i == 2:
                self.play(Create(min_point), Write(min_label))

            self.play(Write(subtitle_obj))
            self.wait(1)
            self.play(FadeOut(subtitle_obj))
        
        # 第四部分：梯度下降迭代过程
        # 梯度下降参数
        learning_rate = 0.3
        start_x = -2.5
        current_x = start_x
        steps = 9
        
        # 创建初始点
        dot = Dot(axes.coords_to_point(current_x, func(current_x)), color=RED)
        dot_label = Text("初始点", font_size=20).next_to(dot, LEFT)
        
        # 添加迭代过程字幕
        iteration_subtitle = Text("现在开始梯度下降的迭代过程。我们从x = -2.5开始，这是一个人为选择的初始点。", 
                                 font_size=20, color=YELLOW).to_edge(DOWN)
        self.play(Write(iteration_subtitle))
        self.wait(1)
        
        self.play(Create(dot), Write(dot_label))
        self.wait(1)
        self.play(FadeOut(dot_label), FadeOut(iteration_subtitle))
        
        # 创建一个信息区域用于显示文本
        info_box = Rectangle(width=5, height=1.5, fill_color=BLACK, fill_opacity=0.8, stroke_color=WHITE)
        info_box.to_corner(DR)
        
        # 只在第一次迭代时显示详细解释
        explanation_shown = False
        
        # 迭代过程
        for i in range(steps):
            # 计算梯度
            gradient = 2 * current_x + 2
            
            # 显示梯度信息
            step_text = Text(f"第 {i+1} 次迭代", font_size=24).to_edge(UP)
            gradient_text = MathTex(f"f'({round(current_x, 2)}) = {round(gradient, 2)}")
            gradient_text.move_to(info_box.get_center() + UP*0.3)
            
            self.play(Write(step_text))
            self.play(Create(info_box), Write(gradient_text))
            
            # 只在第一次迭代时显示详细解释
            if not explanation_shown:
                explanation_subtitle = Text("在每一步迭代中，我们首先计算当前点的梯度值。", 
                                          font_size=20, color=YELLOW).to_edge(DOWN)
                self.play(Write(explanation_subtitle))
                self.wait(1)
                self.play(FadeOut(explanation_subtitle))
                
                gradient_explanation = Text("对于我们的函数f(x) = x² + 2x + 1，梯度(导数)是f'(x) = 2x + 2。", 
                                          font_size=20, color=YELLOW).to_edge(DOWN)
                self.play(Write(gradient_explanation))
                self.wait(1)
                self.play(FadeOut(gradient_explanation))
                
                explanation_shown = True
            
            # 绘制切线
            tangent_line = axes.get_secant_slope_group(
                x=current_x,
                graph=graph,
                dx=0.01,
                dx_label="",
                dy_label="",
                secant_line_color=YELLOW,
                secant_line_length=4,
            )
            self.play(Create(tangent_line))
            self.wait(1)
            
            # 显示梯度方向指示
            if gradient < 0:
                direction_text = Text("负梯度: 增加 x", font_size=20, color=YELLOW)
            else:
                direction_text = Text("正梯度: 减小 x", font_size=20, color=YELLOW)
            
            direction_text.move_to(info_box.get_center() + DOWN*0.2)
            self.play(Write(direction_text))
            self.wait(1)
            
            # 计算下一个点
            next_x = current_x - learning_rate * gradient
            next_dot = Dot(axes.coords_to_point(next_x, func(next_x)), color=RED)
            
            # 显示更新公式
            update_text = MathTex(f"x_{{{i+1}}} = {round(current_x, 2)} - {learning_rate} \\times {round(gradient, 2)} = {round(next_x, 2)}")
            update_text.scale(0.8)
            update_text.next_to(info_box, UP, buff=0.2)
            
            # 只在第一次迭代时显示更新公式的详细解释
            if i == 0:
                update_explanation = Text("然后我们按照更新公式计算下一个点：新位置 = 当前位置 - 学习率 × 梯度", 
                                        font_size=20, color=YELLOW).to_edge(DOWN)
                self.play(Write(update_explanation))
                self.wait(1)
                self.play(FadeOut(update_explanation))
                
                learning_rate_explanation = Text(f"我们使用学习率α = {learning_rate}，这是一个适中的值，既能保证收敛，又不会导致振荡。", 
                                               font_size=20, color=YELLOW).to_edge(DOWN)
                self.play(Write(learning_rate_explanation))
                self.wait(1)
                self.play(FadeOut(learning_rate_explanation))
            
            self.play(Write(update_text))
            self.wait(1)
            
            # 移动到下一个点
            self.play(
                Transform(dot, next_dot),
                FadeOut(tangent_line),
                FadeOut(gradient_text),
                FadeOut(direction_text),
                FadeOut(update_text),
                FadeOut(info_box),
                FadeOut(step_text)
            )
            
            # 只在最后一次迭代前显示继续字幕
            if i == steps - 2:
                continue_subtitle = Text("重复这个过程，我们可以看到点逐渐向最小值移动。", 
                                        font_size=20, color=YELLOW).to_edge(DOWN)
                self.play(Write(continue_subtitle))
                self.wait(1)
                self.play(FadeOut(continue_subtitle))
            
            current_x = next_x
        
        # 第五部分：收敛和分析
        result_text = Text(f"最终结果: x ≈ {round(current_x, 2)}, f(x) ≈ {round(func(current_x), 2)}", font_size=24).to_edge(DOWN)
        exact_text = Text(f"精确解: x = -1, f(x) = 0", font_size=24).next_to(result_text, DOWN)
        
        # 添加收敛字幕
        convergence_subtitle = Text("经过几次迭代后，我们的点已经非常接近理论最小值x = -1。", 
                                  font_size=20, color=YELLOW).to_edge(DOWN)
        self.play(Write(convergence_subtitle))
        self.wait(1)
        self.play(FadeOut(convergence_subtitle))

        self.play(Write(result_text))
        self.wait(1)
        self.play(Write(exact_text))
        self.wait(1)
        
        # 显示学习率的重要性
        # learning_rate_text = Text("学习率的影响:", font_size=24).to_edge(UP)
        # self.play(Write(learning_rate_text))
        self.play(
            FadeOut(axes),
            FadeOut(axes_labels),
            FadeOut(graph),
            FadeOut(func_label),
            FadeOut(min_point),
            FadeOut(min_label),
            FadeOut(dot),
            FadeOut(result_text),
            FadeOut(exact_text)
        )
        # 添加学习率分析字幕
        learning_rate_subtitle = Text("让我们分析一下梯度下降法的特点：", 
                                    font_size=20, color=YELLOW).to_edge(UP)
        self.play(Write(learning_rate_subtitle))
        self.wait(1)
        self.play(FadeOut(learning_rate_subtitle))
        
        # 使用VGroup组织文本，避免重叠
        learning_rate_info = VGroup(
            Text("1. 它简单易实现，只需要计算梯度", font_size=20),
            Text("2. 对于凸函数，它能保证收敛到全局最小值", font_size=20),
            Text("3. 学习率的选择很重要：太小会导致收敛慢，太大会导致振荡甚至发散", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(learning_rate_subtitle, DOWN, buff=0.5)
        
        self.play(Write(learning_rate_info))
        self.wait(3)
        self.play(
            FadeOut(learning_rate_info),
        )
        # 添加实际应用字幕
        application_subtitle = Text("在实际应用中，我们通常会设置收敛条件，比如当梯度值小于某个阈值时停止迭代。", 
                                  font_size=20, color=YELLOW).to_edge(DOWN)
        self.play(Write(application_subtitle))
        self.wait(1)
        self.play(FadeOut(application_subtitle))
        
        # 第六部分：总结

        
        summary_title = Text("梯度下降法总结", font_size=36).to_edge(UP)
        points = VGroup(
            Text("• 沿着负梯度方向迭代更新", font_size=24),
            Text("• 简单易实现，只需计算梯度", font_size=24),
            Text("• 学习率选择很重要", font_size=24),
            Text("• 许多高级算法的基础", font_size=24)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(summary_title, DOWN, buff=0.5)
        
        # 添加总结字幕
        self.play(Write(summary_title))
        self.wait(1)
        self.play(Write(points))
        self.wait(1)
        summary_subtitle = Text("总结一下，梯度下降法通过沿着负梯度方向迭代更新参数，逐步逼近函数最小值。", 
                              font_size=20, color=YELLOW).to_edge(DOWN)
        self.play(Write(summary_subtitle))
        self.wait(1)
        self.play(FadeOut(summary_subtitle))
        
        # 添加扩展字幕
        extension_subtitle = Text("在实际机器学习中，我们经常使用它的变体，如随机梯度下降和小批量梯度下降，来处理大规模数据集。", 
                                font_size=20, color=YELLOW).to_edge(DOWN)
        self.play(Write(extension_subtitle))
        self.wait(1)
        self.play(FadeOut(extension_subtitle))
        
        final_subtitle = Text("希望这个可视化帮助你直观理解梯度下降法的工作原理。谢谢观看！", 
                            font_size=20, color=YELLOW).to_edge(DOWN)
        self.play(Write(final_subtitle))
        self.wait(1)
        
        final_text = Text("谢谢观看!", font_size=48)
        self.play(ReplacementTransform(VGroup(summary_title, points, final_subtitle), final_text))
        self.wait(1)