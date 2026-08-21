from __future__ import annotations
import csv
import numpy as np
from pathlib import Path
from minisom import MiniSom
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "watermelon3.0.csv"
FIG_PATH = Path(__file__).resolve().parent / "ex5_8_som.png"

def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读数据中的密度, 含糖量, 好/坏瓜, 编号
    
    Arg:
        path (Path): 数据路径

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: [密度, 含糖量], 好/坏瓜, 编号
    """
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X, y, ids = [], [], []
    for row in rows:
        X.append([float(row["密度"]), float(row["含糖率"])])
        y.append(1.0 if row["好瓜"] == "是" else 0.0)
        ids.append(int(row["编号"]))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(ids)

def fit_som(h, w, X) -> MiniSom:
    """训练网络"""
    som = MiniSom(h, w, 2, sigma=max(h, w)/2, learning_rate=0.5, random_seed=66)
    som.random_weights_init(X)
    som.train_random(X, 200 * len(X))   # 200 epoch × 17
    return som

def draw_mesh(ax, som, X, y, ids, title):
    W = som.get_weights()
    h, w = W.shape[:2]
    for i in range(h):
        ax.plot(W[i, :, 0], W[i, :, 1], color="0.4", lw=1)
    for j in range(w):
        ax.plot(W[:, j, 0], W[:, j, 1], color="0.4", lw=1)
    ax.scatter(W[:, :, 0], W[:, :, 1], c="0.15", s=18, zorder=3)

    good = y > 0.5
    ax.scatter(X[good, 0], X[good, 1], marker="o", c="#2ca02c", s=42, label="good", zorder=4)
    ax.scatter(X[~good, 0], X[~good, 1], marker="x", c="#d62728", s=42, label="bad", zorder=4)
    for k, sid in enumerate(ids):
        ax.annotate(str(int(sid)), (X[k, 0], X[k, 1]),
                    textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel("density"); ax.set_ylabel("sugar")
    ax.set_title(title); ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")

def draw_hit_map(ax, som, X, y, ids, title):
    W = som.get_weights()
    H, Ww = W.shape[:2]
    cells = [[[] for _ in range(Ww)] for _ in range(H)]
    for x, yi, sid in zip(X, y, ids):
        i, j = som.winner(x)
        cells[i][j].append((int(sid), float(yi)))

    bg = np.zeros((H, Ww))
    for i in range(H):
        for j in range(Ww):
            labs = [yi for _, yi in cells[i][j]]
            if not labs:
                bg[i, j] = 0          # 白：空
            elif all(v > 0.5 for v in labs):
                bg[i, j] = 1          # 绿：纯好瓜
            elif all(v < 0.5 for v in labs):
                bg[i, j] = 2          # 红：纯坏瓜
            else:
                bg[i, j] = 3          # 黄：混

    cmap = ListedColormap(["#ffffff", "#c6efce", "#ffc7ce", "#ffe599"])
    ax.imshow(bg, cmap=cmap, vmin=0, vmax=3, origin="upper")
    for i in range(H):
        for j in range(Ww):
            if not cells[i][j]:
                continue
            txt = ",".join(str(sid) for sid, _ in cells[i][j])
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(Ww)); ax.set_yticks(range(H))
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_title(title)

def main() -> None:

    X, y, ids = load_data(CSV_PATH)
    som33 = fit_som(3, 3, X)
    som44 = fit_som(4, 4, X)

    print(som33.quantization_error(X), som33.topographic_error(X))
    print(som44.quantization_error(X), som44.topographic_error(X))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.5))
    draw_mesh(axes[0, 0], som33, X, y, ids, "3x3 SOM — density-sugar plane")
    draw_hit_map(axes[0, 1], som33, X, y, ids, "3x3 hit map  (green=good, red=bad, yellow=mixed)")
    draw_mesh(axes[1, 0], som44, X, y, ids, "4x4 SOM — density-sugar plane")
    draw_hit_map(axes[1, 1], som44, X, y, ids, "4x4 hit map  (green=good, red=bad, yellow=mixed)")
    fig.suptitle("Watermelon 3.0α — Self-Organizing Map (labels unused in training)")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=140)
    


if __name__ == "__main__":
    main()   