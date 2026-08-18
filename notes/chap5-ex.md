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