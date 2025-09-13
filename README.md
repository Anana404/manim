# Manim 数学优化算法可视化项目

## 项目简介

本项目使用 **Manim** 库创建数学优化算法的可视化演示，旨在通过动画形式深入理解各种优化算法的工作原理。Manim 是一个由 3Blue1Brown 的创作者 Grant Sanderson 开发的 Python 框架，专门用于生成精确的数学动画和教育视频。

## 团队成员

**组长**: 尹子璇  
**组员**: 韩越、边俣、田晔、阙子菁

### 分工安排

- **Adam 优化算法**: 阙子菁
- **梯度下降法**: 韩越
- **共轭梯度法**: 尹子璇
- **牛顿法**: 边俣
- **拉格朗日乘数法**: 田晔

## 项目结构

```
manim/
├── Adam/                          # Adam 优化算法
│   └── Adam-evolution.py         # Adam 算法演进过程可视化
├── Conjugate Gradients/          # 共轭梯度法
│   ├── ConjugateDirections.py    # 共轭方向的几何意义
│   └── ConjugateGradientsProgress.py  # 共轭梯度法算法流程
├── GradientDescent/             # 梯度下降法
│   └── gradient_descent.py      # 梯度下降法基础可视化
├── LagrangeMultipliers/          # 拉格朗日乘数法
│   └── LagrangeDemo.py          # 拉格朗日乘数法几何解释
└── Newton/                      # 牛顿法
    ├── newton_extremum_manim.py # 牛顿法求一维极值
    ├── newton_flat_valley.py    # 牛顿法在平坦谷底的收敛
    └── newton_nonconvex_multiinit.py  # 牛顿法在非凸函数上的多初值对比
```

## 算法覆盖范围

### 1. 梯度下降法 (Gradient Descent)

- **负责人**: 韩越
- **文件**: `GradientDescent/gradient_descent.py`
- **内容**:
  - 梯度下降法的数学原理
  - 迭代过程的可视化演示
  - 学习率对收敛的影响分析
  - 实际应用场景介绍

### 2. Adam 优化算法

- **负责人**: 阙子菁
- **文件**: `Adam/Adam-evolution.py`
- **内容**:
  - Momentum 算法原理和可视化
  - RMSProp 算法原理和可视化
  - Adam 算法的组合优势
  - 算法性能对比分析
  - 3D 可视化应用示例

### 3. 共轭梯度法 (Conjugate Gradient)

- **负责人**: 尹子璇
- **文件**:
  - `Conjugate Gradients/ConjugateDirections.py`
  - `Conjugate Gradients/ConjugateGradientsProgress.py`
- **内容**:
  - 共轭方向的几何意义
  - 最速下降法的"锯齿"路径问题
  - 共轭梯度法的迭代过程
  - 算法步骤的详细解释

### 4. 拉格朗日乘数法 (Lagrange Multipliers)

- **负责人**: 田晔
- **文件**: `LagrangeMultipliers/LagrangeDemo.py`
- **内容**:
  - 约束优化问题的几何解释
  - 等高线与约束曲线的相切条件
  - 拉格朗日函数的构造
  - 数学推导过程

### 5. 牛顿法 (Newton's Method)

- **负责人**: 边俣
- **文件**:
  - `Newton/newton_extremum_manim.py`
  - `Newton/newton_flat_valley.py`
  - `Newton/newton_nonconvex_multiinit.py`
- **内容**:
  - 牛顿法求一维极值的原理
  - 二次近似和泰勒展开
  - 平坦谷底情况下的收敛分析
  - 非凸函数上的多初值对比

## 技术特点

### 可视化特色

- **交互式动画**: 每个算法都包含详细的步骤动画
- **数学公式展示**: 实时显示算法中的关键公式和数值
- **几何直观**: 通过图形化方式解释抽象的数学概念
- **对比分析**: 不同算法的性能对比和适用场景分析

### 代码特点

- **模块化设计**: 每个算法独立成文件，便于理解和修改
- **参数可调**: 支持调整学习率、迭代次数等关键参数
- **错误处理**: 包含数值稳定性处理，如阻尼机制
- **详细注释**: 代码中包含丰富的注释和说明

## 运行环境

### 依赖库

```bash
pip install manim
pip install numpy
pip install sympy
```
