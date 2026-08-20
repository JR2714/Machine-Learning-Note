
from __future__ import annotations # 推迟对类型注解的求值，允许在类定义中使用类本身作为类型提示。

import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import ex5_5_bp as bp

ROOT = Path(__file__).resolve().parent.parent
WINE_PATH = ROOT / "data" / "wine" / "wine.data"
WDBC_PATH = ROOT / "data" / "breast+cancer+wisconsin+diagnostic" / "wdbc.data"
FIG_PATH = Path(__file__).resolve().parent / "ex5_6_curve.png"

# --------------------加载数据-----------------------------
def load_wine(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """加载 Wine 数据集。

    Args:
        path (Path): 数据集路径。

    Returns:
        tuple[np.ndarray, np.ndarray]: 特征矩阵 X 和标签向量 y。
    """
    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))

    X, y = [], []
    label_classes = ["1", "2", "3"]
    for row in rows:
        X.append([float(x) for x in row[1:]])
        y.append([1.0 if label == row[0] else 0.0 for label in label_classes])

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    return X_arr, y_arr

def load_wdbc(path:Path) -> tuple[np.ndarray, np.ndarray]:
    """加载 WDBC 数据集。

    Args:
        path (Path): 数据集路径。

    Returns:
        tuple[np.ndarray, np.ndarray]: 特征矩阵 X 和标签向量 y。
    """
    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))

    X, y = [], []
    for row in rows:
        X.append([float(x) for x in row[2:]])
        y.append([1.0 if row[1] == "M" else 0.0])

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    return X_arr, y_arr

def z_score(X: np.ndarray) -> np.ndarray:
    """对特征矩阵 X 进行 Z-score 标准化。

    Args:
        X (np.ndarray): 特征矩阵。

    Returns:
        np.ndarray: 标准化后的特征矩阵。
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)
# ----------------------------------------------------------

# -------------------- 标准 BP (固定学习率) -----------------------------
def train_standard_bp(
        nn: bp.SingleHiddenBP,
        X: np.ndarray,  
        y: np.ndarray,
        learning_rate: float,
        epochs: int,
        rng: np.random.Generator
    ) -> list[float]:
    """训练标准 BP 神经网络。

    Args:
        nn (bp.SingleHiddenBP): 要训练的神经网络实例.
        X (np.ndarray): 特征矩阵.
        y (np.ndarray): 标签向量.
        learning_rate (float): 学习率.
        epochs (int): 迭代次数.
        rng (np.random.Generator): 随机数生成器.

    Returns:
        list[float]: 每次迭代的损失值列表.
    """
    history: list[float] = [nn.evaluate(X, y)[0]]
    order = np.arange(X.shape[0])
    for _ in range(epochs):
        rng.shuffle(order)  # 打乱顺序，抵消更明显.
        for k in order:
            nn.apply(nn.deltas(X[k], y[k], learning_rate))
        E, _ = nn.evaluate(X, y)
        history.append(E)

    return history

# ----------------------------------------------------------------

# --------------------------- 学习率动态更新器 ----------------------
def learning_rate_update(
        learning_rate: float,
        cur_err: float,
        last_err: float
) -> tuple[float, bool]:
    """更新学习率

    Args:
        learning_rate (float): 当前学习率.
        cur_err (float): 本步损失值.
        last_err (float): 上一步损失值.
    
    Returns:
        float: 更新后的学习率.
        bool: 是否需要回退.
    """
    if cur_err < last_err:
        return learning_rate * 1.1, False
    else:
        return learning_rate * 0.5, True

# --------------------------- 动态学习率 BP 算法 -------------------
def train_dynamic_lr_bp(
        nn: bp.SingleHiddenBP,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        epochs: int,
        rng: np.random.Generator
) -> list[float]:
    """训练神经网络
    
    Args:
        nn (bp.SingleHiddenBP): 要训练的神经网络实例.
        X (np.ndarray): 特征矩阵.
        y (np.ndarray): 标签向量.
        learning_rate (float): 学习率.
        epochs (int): 迭代次数.
        rng (np.random.Generator): 随机数生成器.
        
    Returns:
        list[float]: 每次迭代的损失值列表.
    """
    history: list[float] = [nn.evaluate(X, y)[0]]
    order = np.arange(X.shape[0])
    for _ in range(epochs):
        snap = nn.snapshot()
        rng.shuffle(order)  # 打乱顺序，抵消更明显.
        for k in order:
            nn.apply(nn.deltas(X[k], y[k], learning_rate))
        E, _ = nn.evaluate(X, y)
        history.append(E)
        learning_rate, retro = learning_rate_update(learning_rate, history[-1], history[-2])
        if retro:
            nn.load(snap)
            history[-1] = history[-2]

    return history
# ----------------------------------------------------------------------------

# ----------------------------- 主函数 ----------------------------------------
def main() -> None:
    # 加载数据集
        X_wine, y_wine = load_wine(WINE_PATH)
        X_wdbc, y_wdbc = load_wdbc(WDBC_PATH)
    
        # 对特征进行 Z-score 标准化
        X_wine = z_score(X_wine)
        X_wdbc = z_score(X_wdbc)
    
        # 设置随机数生成器
        rng = np.random.default_rng(0)
    
        # 初始化神经网络
        nn_wine = bp.SingleHiddenBP(d=X_wine.shape[1], q=5, l=y_wine.shape[1], rng=rng)
        nn_wdbc = bp.SingleHiddenBP(d=X_wdbc.shape[1], q=5, l=y_wdbc.shape[1], rng=rng)
        epochs = 100
    
        # 记录初始权重, 让动态学习率的训练可以从相同的初始权重开始
        init_wine = nn_wine.snapshot()
        init_wdbc = nn_wdbc.snapshot()
    
        # 训练标准 BP 神经网络
        wine_history_fixed = train_standard_bp(nn_wine, X_wine, y_wine, learning_rate=0.01, epochs=epochs, rng=rng)
        wdbc_history_fixed = train_standard_bp(nn_wdbc, X_wdbc, y_wdbc, learning_rate=0.01, epochs=epochs, rng=rng)
    
        # 输出训练结果
        print("Wine 数据集训练完成，最终损失值:", wine_history_fixed[-1])
        print("WDBC 数据集训练完成，最终损失值:", wdbc_history_fixed[-1])
    
        # 初始化网络, 让动态学习率为同一起点
        nn_wine.load(init_wine)
        nn_wdbc.load(init_wdbc)
    
        # 训练动态学习率 BP 神经网络
        wine_history_dynamic = train_dynamic_lr_bp(nn_wine, X_wine, y_wine, learning_rate=0.1, epochs=epochs, rng=rng)
        wdbc_history_dynamic = train_dynamic_lr_bp(nn_wdbc, X_wdbc, y_wdbc, learning_rate=0.1, epochs=epochs, rng=rng)
    
        # 输出训练结果
        print("动态学习率 Wine 数据集训练完成，最终损失值:", wine_history_dynamic[-1])
        print("动态学习率 WDBC 数据集训练完成，最终损失值:", wdbc_history_dynamic[-1])

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=False)

        axes[0].plot(wine_history_fixed, label="fixed learning rate", linewidth=1.2)
        axes[0].plot(wine_history_dynamic, label="dynamic learning rate", linewidth=1.2)
        axes[0].set_title("Wine")

        axes[1].plot(wdbc_history_fixed, label="fixed learning rate", linewidth=1.2)
        axes[1].plot(wdbc_history_dynamic, label="dynamic learning rate", linewidth=1.2)
        axes[1].set_title("WDBC")

        for ax in axes:
            ax.set_xlabel("epoch")
            ax.legend()
        axes[0].set_ylabel("E  (eq. 5.16)")

        fig.suptitle("standard BP: fixed vs dynamic learning rate (same init)")
        fig.tight_layout()
        fig.savefig(FIG_PATH, dpi=140)
        print("曲线已保存: ", FIG_PATH)
# ------------------------------------------------------------------------------------

if __name__ == "__main__":
   main()