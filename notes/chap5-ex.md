# Chap5 Exercises

## 5.1 试述将线性函数 $f(\mathbf{x})=\omega^{T}\mathbf{x}$ 用作神经元激活函数的缺陷.

失去了非线性的特点, 导致无论多少层网络本质上只是一个线性函数: 即使每层带有阈值, 仿射映射的复合仍是仿射,
$$
f^{(L)}\circ\cdots\circ f^{(1)}(\mathbf{x})=W'\mathbf{x}+\mathbf{b}',
$$
深度不增加任何表达能力, 网络退化成单层线性模型, 无法解决异或等非线性可分问题. 此外线性激活输出无界, 没有 Sigmoid 那种挤压, 多层时数值也更容易爆炸.

## 5.2 试述使用图 5.2(b) 激活函数的神经元与对率回归的联系.

补看 $Chap3\,\S3.3$ 后, 我认为使用 Sigmoid 函数作为激活函数的神经元本质上是一个最简单的对率回归器: 单个神经元
$$
y=\bigl(1+\exp(-(\boldsymbol{\omega}^{\mathrm{T}}\mathbf{x}-\theta))\bigr)^{-1}
$$
与对数几率回归是同一个函数. 对于输入 $\boldsymbol{\omega}^{\mathrm{T}}\mathbf{x}+b$, Sigmoid 将其压缩到 $(0,1)$, 估量该神经节点应当活跃的"可能性". 差别主要在由来与惯用训练: 对率回归从 $\ln\frac{y}{1-y}$ 线性、Bernoulli 极大似然(交叉熵)出发; 神经元从 M-P + 可微激活出发, 本书 BP 用的是均方误差. 函数形式相同, 优化目标可以不同.

## 5.3 对于图 5.7 中的 $v_{ih}$ 试推导出 BP 算法中的更新公式(5.13).

由于不涉及需要使用矩阵简写运算的情况, 我们采用回书本的下标写法.

需求: $\Delta v_{ih}=-\eta \frac{\partial E_k}{\partial v_{ih}}$.

分析链路: 
$$
\begin{aligned}
\color{red}{\alpha_h}&=\sum_{i'=1,i'\neq i}^dv_{i'h}x_{i'}+\textcolor{red}{v_{ih}} x_i\\
\color{red}{b_h}&=f(\textcolor{red}{\alpha_h} -\gamma_h)\\
\textcolor{red}{\beta_j}&=\sum_{p=1, p\neq h}^q\omega_{pj}b_p+\omega_{hj}\textcolor{red}{b_h}\\
\textcolor{red}{\hat{y}_j^k}&=f(\textcolor{red}{\beta_j}-\theta_j)\\
\textcolor{red}{E_k}&=\frac{1}{2}\sum_{j=1}^l(\textcolor{red}{\hat{y}_j^k}-y_j^k)^2
\end{aligned}
$$

因此 (并用上 $f'(x)=f(x)(1-f(x))$):
$$
\begin{aligned}
\frac{\partial E_k}{\partial v_{ih}}&=\sum_{j=1}^l\frac{\partial E_k}{\partial \hat{y}_j^k}\cdot\frac{\partial \hat{y}_j^k}{\partial \beta_j}\cdot\frac{\partial \beta_j}{\partial b_h}\cdot\frac{\partial b_h}{\partial\alpha_h}\cdot\frac{\partial \alpha_h}{\partial v_{ih}}\\
&=\sum_{j=1}^l(\hat{y}_j^k-y_j^k)f'(\beta_j-\theta_j)\omega_{hj}f'(\alpha_h-\gamma_h)x_i\\
&=\sum_{j=1}^l(\hat{y}_j^k-y_j^k)\hat{y}_j^k(1-\hat{y}_j^k)\omega_{hj}b_h(1-b_h)x_i
\end{aligned}
$$
记
$$
\begin{aligned}
g_j&=-(\hat{y}_j^k-y_j^k)\hat{y}_j^k(1-\hat{y}_j^k)=\hat{y}_j^k(1-\hat{y}_j^k)(y_j^k-\hat{y}_j^k)\\
e_h&=b_h(1-b_h)\sum_{j=1}^l\omega_{hj} g_j
\end{aligned}
$$
则 $\partial E_k/\partial v_{ih}=(-e_h)x_i$, 从而
$$
\Delta v_{ih}=-\eta\frac{\partial E_k}{\partial v_{ih}}=\eta e_hx_i.
$$

## 5.4 试述式(5.6) 中学习率的取值对神经网络训练的影响.

学习率 $\eta\in(0,1)$ 控制着算法每一轮迭代中的更新步长. $\eta$ 太大则容易振荡: 一步跨过极小点, 参数在谷两侧来回跳, 甚至发散; 太小则收敛速度又会过慢, 也更容易停在较差的局部极小. 有时为了做精细调节, 可令式(5.11) 与 (5.12) 使用 $\eta_1$, 式(5.13) 与 (5.14) 使用 $\eta_2$, 两者未必相等——输出层梯度 $g_j$ 与隐层梯度 $e_h$ 量级常常不同 (Sigmoid 导数 $\le 1/4$, 隐层还要再乘一层), 分开调更稳. 实践中也常先大后小.

## 5.5 试编程实现标准 BP 算法和累积 BP 算法?在西瓜数据集 3.0 上分别用这两个算法训练一个单隐层网络，并进行比较.

[代码](../exercises/ex5_5_bp.py); [无序结果](../exercises/ex5_5_curve.png), [有序结果](../exercises/ex5_5_encoding.png).

通过训练和修改 p (隐藏层神经元数) 以及 sed (不同参数初始值), 训练结果都指向标准 BP 收敛的速度更快. 有序化降了 d，本题里没让优化更好；标准仍快于累积.

## 5.6 试设计一个 BP 改进算法，能通过动态调整学习率显著提升收敛速度.编程实现该算法，并选择两个 UCI 数据集与标准 BP 算法进行实验比较.

本题选择了 Wine（178×13，3 类）和 Breast Cancer Wisconsin Diagnostic（WDBC，569×30，2 类）作为实验数据集。特征做 z-score（Wine 各列尺度差很大，不标准化则比的是尺度而不是 $\eta$）；Wine 标签 one-hot，$l=3$，WDBC 二分类 $l=1$。网络复用 5.5 的 `SingleHiddenBP`，$q=5$，两种算法同一 `snapshot` 初值，只改 $\eta$ 策略。固定对照 $\eta=0.01$，动态版起步 $0.1$；100 epoch，全程看训练 $E$（式 5.16）。本题比的是收敛速度，暂不划分训练/测试。

算法是 Bold Driver（阅读材料里的先大后小）：$E$ 下降则 $\eta\leftarrow\eta\times 1.1$（或消融里的 $\times 2$），$E$ 上升则参数回退到本 epoch 之前，损失记录一并改回（`history[-1]=history[-2]`），$\eta\leftarrow\eta\times 0.5$。

最初没有回退，且增长过猛（$\times 2$）：前几步明显快于固定 $\eta$，随后剧烈抖动，并常常停在比标准 BP 更高的 $E$。加上回退后，$\times 2$ 不再停在高台，但仍有齿状；正式采用 $\times 1.1/\times 0.5$ + 回退，两数据集上都显著快于固定 $\eta$，且曲线稳定。

[代码](../exercises/ex5_6.py)（网络见 [5.5](../exercises/ex5_5_bp.py) 的 `SingleHiddenBP`）；[×2 无回退](../exercises/ex5_6_curve_100ep_x2x0.5.png)；[×2 有回退](../exercises/ex5_6_curve_100ep_x2x0.5_with_revert.png)；[×1.1 有回退（主结论）](../exercises/ex5_6_curve_100ep_x1.1x0.5_with_revert.png)。

## 5.7 根据式(5.18) (5.19) ，试构造一个能解决异或问题的单层 RBF 神经网络.

异或: $(0,0)\to 0$, $(0,1)\to 1$, $(1,0)\to 1$, $(1,1)\to 0$. RBF 要解决 XOR, 需要 "分清楚" 两条对角线——XOR $=1$ 的 $(1,0),(0,1)$ 和 XOR $=0$ 的 $(0,0),(1,1)$, 即网络在两类对角线上的输出要有明显大小区分.

选用 2 个隐元, 中心放在 1 类对角 $(1,0)$, $(0,1)$ (两座山), 高斯径向基 (5.19), $\beta>0$ 任取 (下面取 $\beta=1$). 式 (5.18) 无显式偏置, 加一个恒为 $1$ 的基, 输出
$$
\varphi(\mathbf{x})=w_1\rho(\mathbf{x},\mathbf{c}_1)+w_2\rho(\mathbf{x},\mathbf{c}_2)+\theta.
$$
四个顶点给出 $4\times 3$ 线性方程组, 未知数 $(w_1,w_2,\theta)$. $(0,0)$ 与 $(1,1)$ 到两座山的距离都是 $1$, 这两行相同, 实际只有三条独立方程, 恰好可解. 对称闭式:
$$
w_1=w_2=\frac{1}{(1-e^{-\beta})^2},\qquad
\theta=-\frac{2e^{-\beta}}{(1-e^{-\beta})^2}.
$$
$\beta=1$ 时 $w\approx 2.50$, $\theta\approx -1.84$, 四个顶点精确 $0/1$.

本题是表示不是学习: 中心和 $\beta$ 是按标签几何定的, $W$ 是四个点上的线性最小二乘, 没有未见样本.

[代码示例](../exercises/rbf_net.py)