from manim import *
import numpy as np

class Test(Scene):
    def construct(self):
        axes=NumberPlane(
            axis_config={"stroke_color":GRAY,"stroke_width":0.5},
            background_line_style={"stroke_color":GRAY,"stroke_width":0.2}).add_coordinates()
        self.add(axes)

        # 双曲线
        def f(x):
            return 1/x
        graph1 = axes.plot(f,x_range=[-4,-0.3],color=RED,stroke_width=3)
        graph2 = axes.plot(f,x_range=[0.3,4],color=RED,stroke_width=3)
        label = MathTex(r"y=\frac{1}{x}",font_size=35,color=RED).move_to(axes.c2p(-0.2,3.3))
        self.play(Create(graph1),Create(graph2),Create(label))
        self.wait(1)
        
        t1 = Text("求曲线上的点到原点的最小距离?",font_size=26,gradient=(GOLD,BLUE,RED)).move_to(axes.c2p(-4,3))
        self.play(Write(t1))
        self.wait(1)

        # 曲线上的点到原点距离的运动演示
        dot1 = Dot(axes.c2p(-4,-1/4),color=WHITE)
        self.play(FadeIn(dot1))
        line1 = Line(ORIGIN,dot1,color=YELLOW)
        #self.add(line1)
        self.play(
            MoveAlongPath(dot1, graph1),
            UpdateFromFunc(line1, lambda l: l.put_start_and_end_on(ORIGIN, dot1.get_center())),
            rate_func=linear,
            run_time=3
        )
        self.play(FadeOut(line1, dot1))
        dot2 = Dot(axes.c2p(0.3,1/0.3),color=WHITE)
        self.play(FadeIn(dot2))
        line2 = Line(ORIGIN,dot2,color=YELLOW)
        #self.add(line2)
        self.play(
            MoveAlongPath(dot2, graph2),
            UpdateFromFunc(line2, lambda l: l.put_start_and_end_on(ORIGIN, dot2.get_center())),
            rate_func=linear,
            run_time=4
        )

        t2 = Text("1、哪条线段最短呢？",font_size=24,color=WHITE).move_to(axes.c2p(-5,2))
        self.play(FadeIn(t2))
        self.wait(2)
        self.play(FadeOut(line2, dot2))

        # 画圆：等高线
        for i in np.arange(0.2,2.2,0.2):
            c = Circle(radius=i,color=BLUE)
            self.play(FadeIn(c))

        t3 = Text("2、等高线圆,关注曲线相切圆",font_size=24,color=YELLOW).next_to(t2,DOWN*2.2).shift(RIGHT*0.65)
        self.play(FadeIn(t3))

        # 切线圆
        circle = Circle(radius=np.sqrt(2),color=YELLOW)
        self.play(Write(circle))
        self.wait(2)
        # 最短距离
        line3 = Line(ORIGIN,axes.c2p(1,1),color=PINK)
        line4 = Line(ORIGIN,axes.c2p(-1,-1),color=GREEN)
        self.play(Create(line3),Create(line4))
        self.wait(2)
        # 曲线等高线
        for i in np.arange(0.5,3,0.5):
            graph = axes.plot(lambda x: 1/(x-i)+i,x_range=[0.3+i,4+i],color=RED,stroke_width=3)
            self.play(Create(graph))
        
        t4 = Text("3、同样画出曲线的等高线",font_size=24,color=RED).next_to(t3,DOWN*2.2).shift(LEFT*0.2)
        self.play(FadeIn(t4))
        self.wait(2)
        t5 = Text("4、其法线方向是梯度走向",font_size=24,color=WHITE).next_to(t4,DOWN*2.2)
        self.play(FadeIn(t5))
        self.wait(2)
        #切线
        dl = DashedLine(axes.c2p(-4,6),axes.c2p(7,-5),color=WHITE)
        self.play(Create(dl))
        self.wait(2)

        t6 = Text("5、共同切线的法线共线",font_size=24,color=YELLOW).next_to(t5,DOWN*2.2).shift(LEFT*0.15)
        self.play(FadeIn(t6))
        self.wait(1)

        # 法线向量
        arrow1 = Arrow(axes.c2p(1,1),axes.c2p(3,3),buff=0,color=RED)
        arrow2 = Arrow(axes.c2p(1,1),axes.c2p(-2,-2),buff=0,color=BLUE)
        self.play(GrowArrow(arrow1),GrowArrow(arrow2))
        self.wait(1)

        t7 = Text("于是拉格朗日乘数法，表示成如下线性组合：",font_size=30,color=YELLOW).move_to(axes.c2p(-3,-2.5))
        self.play(FadeIn(t7))
        self.wait(1)

        t8 = MathTex(r"L(x,y,\lambda) = f(x,y) + \lambda \cdot g(x,y)",font_size=40,color=RED).move_to(axes.c2p(0,-3.3))
        self.play(FadeIn(t8))
        self.wait(1)
        self.play(t8.animate.scale(1.5))
        self.wait(3)
        
        # 隐藏说明信息
        vg = VGroup(t1,t2,t3,t4,t5,t6,t7)
        self.play(Unwrite(vg))
        self.play(t8.animate.move_to(axes.c2p(-4,3)).scale(0.6))
        self.wait(2)
        sr = SurroundingRectangle(t8,corner_radius=0.1)
        self.play(Create(sr))
        self.wait(2)

        t8_info = Text("(f是目标函数 g是约束条件 λ是拉格朗日乘数)",font_size=20,color=YELLOW).next_to(t8,DOWN)
        self.play(FadeIn(t8_info))
        self.wait(3)
        g1 = MathTex(r"L(x,y,\lambda) = (x^2+y^2) + \lambda \cdot (xy-1)",font_size=35,color=RED).next_to(t8_info,DOWN)
        self.play(FadeIn(g1))
        self.wait(2)
        t9 = Text("故极值点是各自偏导为0",font_size=24,color=YELLOW).next_to(g1,DOWN)
        self.play(FadeIn(t9))
        self.wait(2)
    # 数学推导过程
        # 隐藏前面的说明文字
        self.play(Unwrite(sr),Unwrite(t8),Unwrite(t8_info), Unwrite(g1), Unwrite(t9))
        # 数学推导过程字体减小
        math_group = VGroup(
            Text("目标函数:", font_size=18, color=WHITE),
            MathTex(r"f(x, y) = x^2 + y^2", font_size=20, color=WHITE),
            Text("约束条件:", font_size=18, color=WHITE),
            MathTex(r"g(x, y) = xy - 1 = 0", font_size=20, color=WHITE),
            Text("构造拉格朗日函数:", font_size=18, color=WHITE),
            MathTex(r"L(x, y, \lambda) = x^2 + y^2 + \lambda(xy - 1)", font_size=20, color=WHITE),
            Text("令各偏导为零:", font_size=18, color=YELLOW),
            MathTex(r"\frac{\partial L}{\partial x} = 2x + \lambda y = 0", font_size=20, color=YELLOW),
            MathTex(r"\frac{\partial L}{\partial y} = 2y + \lambda x = 0", font_size=20, color=YELLOW),
            MathTex(r"\frac{\partial L}{\partial \lambda} = xy - 1 = 0", font_size=20, color=YELLOW),
            Text("由第三式:", font_size=18, color=WHITE),
            MathTex(r"xy = 1", font_size=20, color=WHITE),
            Text("由第一、二式:", font_size=18, color=WHITE),
            MathTex(r"2x + \lambda y = 0,\ 2y + \lambda x = 0", font_size=20, color=WHITE),
            Text("联立解得:", font_size=18, color=WHITE),
            MathTex(r"x = y = \pm 1", font_size=20, color=WHITE),
            Text("物理意义：极值点处，目标函数和约束曲线的切线方向一致，即梯度共线。", font_size=18, color=GOLD)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(DOWN, buff=0)
        self.play(FadeIn(math_group))
        self.wait(8)
        self.play(Unwrite(math_group))
        L1 = MathTex(r"L'_x = 2x + \lambda y = 0",font_size=35,color=WHITE).next_to(t9,DOWN)
        L2 = MathTex(r"L'_y = 2y + \lambda x = 0",font_size=35,color=WHITE).next_to(L1,DOWN)
        L3 = MathTex(r"L'_\lambda = xy - 1 = 0",font_size=35,color=WHITE).next_to(L2,DOWN).shift(LEFT*0.1)
        L123 = VGroup(L1,L2,L3)
        brace = Brace(L123,LEFT)
        self.play(FadeIn(L123,brace))
        self.wait(5)
        L4 = MathTex(r"\boldsymbol{\Longrightarrow}",r"x=y=\pm 1",font_size=40,color=WHITE).next_to(L3,DOWN*2).shift(LEFT*0.5)
        L4[0].set_color(RED)
        self.play(FadeIn(L4))
        self.wait(2)
        L5 = MathTex(r"\boldsymbol{\Longrightarrow}",r"\quad\sqrt{x^2+y^2}",r"\quad = \sqrt{2}",font_size=40,color=WHITE).next_to(L4,DOWN*2)
        L5[0].set_color(RED).shift(RIGHT*0.5)
        self.play(TransformFromCopy(L4,L5))
        self.wait(1)
        self.play(L5[2].animate.scale(1.5).set_color(YELLOW))
        self.wait(5)