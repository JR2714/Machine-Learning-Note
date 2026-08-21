from __future__ import annotations

import numpy as np

XOR_X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)

XOR_Y = np.array([[0.0], [1.0], [1.0], [0.0]])

class rbf_net:
    """用于判断异或问题的 RBF 神经网络。

    2 -> 2 -> 1
    隐层是高斯径向基.
    W为输出权.
    """

    def __init__(self) -> None:
        """初始化网络。

        固定输入维度 2, 隐层维度 2, 输出维度 1.
        """
        self.center = np.array([[1.0, 0.0], [0.0, 1.0]]) # 中心设为一类对角线两座山峰
        self.beta = 1.0 # 高斯径向基的参数
        self.W = np.empty([3, 1]) # 多出来一个维度用来放输出层阈值 theta

    def solve_xor_weight(self) -> None:
        """两步法第二步：中心定死后，解线性方程组定输出权。

        4 个顶点, 3 个未知数 (w1, w2, theta). (0,0) 与 (1,1) 到两座山的
        距离都是 1 → 这两行相同, 实际是 3 个独立方程, 唯一解.
        """
        Phi = np.array(
            [
                self.gaussian(XOR_X[0]),      # (0,0): 到 (1,0)、(0,1) 距离平方都是 1
                self.gaussian(XOR_X[1]),      # (0,1): 到 (1,0) 距离平方=2；到自己=0
                self.gaussian(XOR_X[2]),      # (1,0): 与 (0,1)类似
                self.gaussian(XOR_X[3]),      # (1,1): 与 (0,0) 行相同（0 类两点对称）
            ]
        )
        self.W, _, _, _ = np.linalg.lstsq(Phi, XOR_Y)

    def gaussian(self, x: np.ndarray) -> np.ndarray:
        """式 (5.19)

        Args:
            x (np.ndarray): 单个输入, 形状 (2,)。

        Returns:
            np.ndarray: 两个隐元的径向基输出, 最后一维为阈值, 形状 (3,)
        """
        sq = np.sum((x - self.center) ** 2, axis=1)
        return np.append(np.exp(-self.beta * sq), 1.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """网络运行"""
        return self.gaussian(x) @ self.W


    def test(self) -> None:
        """测试网络"""
        print(f"输出权: {self.W}")
        print(f"(0,0) 结果: {self.forward(np.array([0, 0]))}")
        print(f"(0,1) 结果: {self.forward(np.array([0, 1]))}")
        print(f"(1,0) 结果: {self.forward(np.array([1, 0]))}")
        print(f"(1,1) 结果: {self.forward(np.array([1, 1]))}")
def main() -> None:
    nn = rbf_net()
    nn.solve_xor_weight()
    nn.test()

if __name__ == "__main__":
    main()
