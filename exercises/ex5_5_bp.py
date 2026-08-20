"""西瓜书习题 5.5：标准 BP vs 累积 BP。

对照图 5.8 与式 (5.10)–(5.16)，手写单隐层网络（不用 autograd）。
在西瓜数据集 3.0 上用同一初始化各训一遍，比较累积误差曲线。

运行（在项目根目录 ml-theory/）:

    python exercises/ex5_5_bp.py
"""

from __future__ import annotations

import csv
import copy
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 路径：不依赖你从哪个目录启动，一律相对本文件定位
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "watermelon3.0.csv"
FIG_PATH = Path(__file__).resolve().parent / "ex5_5_curve.png"
FIG_ENC_PATH = Path(__file__).resolve().parent / "ex5_5_encoding.png"


# 全部离散属性 one-hot 时的取值顺序（书表 4.3）。
DISCRETE = {
    "色泽": ["青绿", "乌黑", "浅白"],
    "根蒂": ["蜷缩", "稍蜷", "硬挺"],
    "敲声": ["浊响", "沉闷", "清脆"],
    "纹理": ["清晰", "稍糊", "模糊"],
    "脐部": ["凹陷", "稍凹", "平坦"],
    "触感": ["硬滑", "软粘"],
}

# 混合编码：色泽 / 敲声仍 one-hot；其余按 §3.2 连续化。
# 三值取 1.0 / 0.5 / 0.0，方向按「更像好瓜」的常见偏好排。
ONEHOT_KEEP = {
    "色泽": ["青绿", "乌黑", "浅白"],
    "敲声": ["浊响", "沉闷", "清脆"],
}
ORDINAL = {
    "根蒂": {"蜷缩": 1.0, "稍蜷": 0.5, "硬挺": 0.0},
    "纹理": {"清晰": 1.0, "稍糊": 0.5, "模糊": 0.0},
    "脐部": {"凹陷": 1.0, "稍凹": 0.5, "平坦": 0.0},
    "触感": {"硬滑": 1.0, "软粘": 0.0},
}


def _encode_row(row: dict, encoding: str) -> list[float]:
    feats: list[float] = []
    if encoding == "onehot":
        for col, values in DISCRETE.items():
            feats.extend(1.0 if row[col] == v else 0.0 for v in values)
    elif encoding == "ordinal":
        for col, values in ONEHOT_KEEP.items():
            feats.extend(1.0 if row[col] == v else 0.0 for v in values)
        for col, mapping in ORDINAL.items():
            feats.append(mapping[row[col]])
    else:
        raise ValueError(f"unknown encoding: {encoding}")
    feats.append(float(row["密度"]))
    feats.append(float(row["含糖率"]))
    return feats


def load_watermelon(path: Path, encoding: str = "onehot") -> tuple[np.ndarray, np.ndarray]:
    """读 CSV → X (m, d), y (m, 1)。

    encoding="onehot"   ：全部离散属性 one-hot，§3.2 默认。
    encoding="ordinal"  ：色泽、敲声 one-hot；根蒂/纹理/脐部/触感连续化。
    """
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X, y = [], []
    for row in rows:
        X.append(_encode_row(row, encoding))
        y.append([1.0 if row["好瓜"] == "是" else 0.0])

    X_arr = np.asarray(X, dtype=np.float64)  # 用 asarray 而不是 array 来避免不必要的复制, 提高效率
    y_arr = np.asarray(y, dtype=np.float64)
    return X_arr, y_arr


def sigmoid(z: np.ndarray) -> np.ndarray:
    # clip 只是防止 exp 溢出，不改变书上的函数形式
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


class SingleHiddenBP:
    """图 5.7 的单隐层网络。下标与书一致：v_ih, ω_hj。

    V[i, h]     = v_ih     输入 i → 隐层 h
    gamma[h]    = γ_h      隐层阈值
    W[h, j]     = ω_hj     隐层 h → 输出 j
    theta[j]    = θ_j      输出层阈值
    """

    def __init__(self, d: int, q: int, l: int, rng: np.random.Generator) -> None:
        """图 5.8 第 1 行：权、阈值在 (0, 1) 内随机初始化

        Args:
            d (int): 输入维度。
            q (int): 隐层神经元个数。
            l (int): 输出维度。
            rng (np.random.Generator): 随机数生成器。
        """
        self.V = rng.uniform(0.0, 1.0, size=(d, q))
        self.gamma = rng.uniform(0.0, 1.0, size=(q,))
        self.W = rng.uniform(0.0, 1.0, size=(q, l))
        self.theta = rng.uniform(0.0, 1.0, size=(l,))

    def snapshot(self) -> dict:
        """拷一份参数，给两种算法当同一起点。"""
        return {
            "V": self.V.copy(),
            "gamma": self.gamma.copy(),
            "W": self.W.copy(),
            "theta": self.theta.copy(),
        }

    def load(self, params: dict) -> None:
        self.V = params["V"].copy()
        self.gamma = params["gamma"].copy()
        self.W = params["W"].copy()
        self.theta = params["theta"].copy()

    def forward_one(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """单个样本前向。返回 (yhat, b)，反传要用到隐层输出 b。"""
        alpha = x @ self.V                  # α_h = Σ_i v_ih x_i
        b = sigmoid(alpha - self.gamma)     # b_h = σ(α_h − γ_h)
        beta = b @ self.W                   # β_j = Σ_h ω_hj b_h
        yhat = sigmoid(beta - self.theta)   # ŷ_j = σ(β_j − θ_j)   式(5.3)
        return yhat, b

    def deltas(self, x: np.ndarray, y: np.ndarray, eta: float) -> dict:
        """按式 (5.10)–(5.15) 算出这一份样本的 Δ（已乘 η，尚未加到参数上）。"""
        yhat, b = self.forward_one(x)

        # 式(5.10)：g_j = ŷ_j (1 − ŷ_j) (y_j − ŷ_j)
        g = yhat * (1.0 - yhat) * (y - yhat)
        # 式(5.15)：e_h = b_h (1 − b_h) Σ_j ω_hj g_j
        e = b * (1.0 - b) * (self.W @ g)

        # 式(5.11)–(5.14)
        return {
            "W": eta * np.outer(b, g),       # Δω_hj = η g_j b_h
            "theta": -eta * g,               # Δθ_j  = −η g_j
            "V": eta * np.outer(x, e),       # Δv_ih = η e_h x_i
            "gamma": -eta * e,               # Δγ_h  = −η e_h
        }

    def apply(self, delta: dict) -> None:
        self.V += delta["V"]
        self.gamma += delta["gamma"]
        self.W += delta["W"]
        self.theta += delta["theta"]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """累积误差 E = (1/m) Σ E_k（式 5.16）和训练准确率。"""
        m = X.shape[0]
        se = 0.0
        correct = 0
        for k in range(m):
            yhat, _ = self.forward_one(X[k])
            se += 0.5 * float(np.sum((yhat - y[k]) ** 2))  # E_k，式(5.4)
            pred = 1.0 if yhat[0] > 0.5 else 0.0
            correct += int(pred == y[k, 0])
        return se / m, correct / m

    def train_standard(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eta: float,
        epochs: int,
        rng: np.random.Generator,
    ) -> list[float]:
        """标准 BP = 图 5.8：每个样本算完立刻更新（≈ SGD）。"""
        m = X.shape[0]
        history: list[float] = []
        order = np.arange(m)
        for _ in range(epochs):
            rng.shuffle(order)  # 打乱顺序，抵消更明显；累积 BP 不需要这一步
            for k in order:
                self.apply(self.deltas(X[k], y[k], eta))
            E, _ = self.evaluate(X, y)
            history.append(E)
        return history

    def train_accumulated(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eta: float,
        epochs: int,
    ) -> list[float]:
        """累积 BP：对 E = (1/m) Σ E_k 求梯度，读完一整遍再更新一次。"""
        m = X.shape[0]
        history: list[float] = []
        for _ in range(epochs):
            acc = {key: np.zeros_like(val) for key, val in self.snapshot().items()}
            for k in range(m):
                delta = self.deltas(X[k], y[k], eta)
                for key in acc:
                    acc[key] += delta[key]
            # 除以 m：对应式(5.16) 的 1/m，这样两种算法用同一个 η 才公平
            for key in acc:
                acc[key] /= m
            self.apply(acc)
            E, _ = self.evaluate(X, y)
            history.append(E)
        return history


def first_hit(history: list[float], thresh: float) -> str:
    for t, e in enumerate(history, start=1):
        if e < thresh:
            return str(t)
    return f"> {len(history)}"


def run_pair(
    X: np.ndarray,
    y: np.ndarray,
    q: int,
    eta: float,
    epochs: int,
    seed: int,
) -> tuple[list[float], list[float], tuple[float, float], tuple[float, float]]:
    """同一初始化下跑标准 BP 与累积 BP。d 不同则权形状不同，只能保证同一套随机种子。"""
    d = X.shape[1]
    l = y.shape[1]
    rng = np.random.default_rng(seed)
    net = SingleHiddenBP(d, q, l, rng)
    init = net.snapshot()

    net.load(init)
    hist_std = net.train_standard(X, y, eta, epochs, copy.deepcopy(rng))
    stat_std = net.evaluate(X, y)

    net.load(init)
    hist_acc = net.train_accumulated(X, y, eta, epochs)
    stat_acc = net.evaluate(X, y)
    return hist_std, hist_acc, stat_std, stat_acc


def print_row(name: str, E: float, acc: float, hist: list[float]) -> None:
    print(
        f"{name:<18}{E:10.4f}{acc:12.2%}"
        f"{first_hit(hist, 0.05):>10}{first_hit(hist, 0.02):>10}{first_hit(hist, 0.01):>10}"
    )


def main() -> None:
    q = 5
    eta = 0.5
    epochs = 4000
    seed = 66

    print("有序映射（越大越像常见的好瓜偏好）:")
    for col, mapping in ORDINAL.items():
        shown = ", ".join(f"{k}={v:g}" for k, v in mapping.items())
        print(f"  {col}: {shown}")
    print("仍 one-hot: 色泽, 敲声")
    print()

    results = {}
    for encoding in ("onehot", "ordinal"):
        X, y = load_watermelon(CSV_PATH, encoding=encoding)
        m, d = X.shape
        n_param = (d + y.shape[1] + 1) * q + y.shape[1]
        print(
            f"[{encoding}] m={m}  d={d}  正例={int(y.sum())}  "
            f"网络 {d}-{q}-{y.shape[1]}  参数 {n_param}  η={eta}  seed={seed}"
        )
        results[encoding] = (X, y, *run_pair(X, y, q, eta, epochs, seed))

    print()
    print(f"{'':<18}{'最终 E':>10}{'训练 acc':>12}{'E<0.05':>10}{'E<0.02':>10}{'E<0.01':>10}")
    for encoding, label_std, label_acc in (
        ("onehot", "onehot / 标准", "onehot / 累积"),
        ("ordinal", "ordinal / 标准", "ordinal / 累积"),
    ):
        _, _, hist_std, hist_acc, stat_std, stat_acc = results[encoding]
        print_row(label_std, stat_std[0], stat_std[1], hist_std)
        print_row(label_acc, stat_acc[0], stat_acc[1], hist_acc)

    # 左：原 5.5 对照；右：两种编码叠在一起看有序化差多少
    _, _, hist_std_oh, hist_acc_oh, _, _ = results["onehot"]
    _, _, hist_std_ord, hist_acc_ord, _, _ = results["ordinal"]

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(hist_std_oh, label="standard BP (SGD)", linewidth=1.2)
    plt.plot(hist_acc_oh, label="accumulated BP (batch GD)", linewidth=1.2)
    plt.xlabel("epoch")
    plt.ylabel("E  (eq. 5.16)")
    plt.title("Watermelon 3.0  —  standard vs accumulated BP (one-hot)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=140)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
    axes[0].plot(hist_std_oh, label="standard", linewidth=1.2)
    axes[0].plot(hist_acc_oh, label="accumulated", linewidth=1.2)
    axes[0].set_title(f"one-hot  d={results['onehot'][0].shape[1]}")
    axes[1].plot(hist_std_ord, label="standard", linewidth=1.2)
    axes[1].plot(hist_acc_ord, label="accumulated", linewidth=1.2)
    axes[1].set_title(f"mixed ordinal  d={results['ordinal'][0].shape[1]}")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()
    axes[0].set_ylabel("E  (eq. 5.16)")
    fig.suptitle("Watermelon 3.0  —  encoding ablation (same q / η / seed)")
    fig.tight_layout()
    fig.savefig(FIG_ENC_PATH, dpi=140)
    print(f"\n曲线已保存: {FIG_PATH}")
    print(f"编码对照:   {FIG_ENC_PATH}")


if __name__ == "__main__":
    main()
