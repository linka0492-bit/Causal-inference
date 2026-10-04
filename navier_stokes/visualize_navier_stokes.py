"""
Две визуализации:
  1) дерево/DAG уравнения (в стиле visualize_tree.py, но с учётом
     NONLOCAL-вершин давления -- выделены отдельным цветом/формой);
  2) сама физика -- поле завихренности в несколько моментов времени,
     чтобы увидеть характерную неустойчивость Кельвина-Гельмгольца.
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

from navier_stokes_tree import build_navier_stokes_tree, PDEExpressionTree, NodeColor
from navier_stokes_solver import NavierStokes2D, make_shear_layer_ic

COLOR_MAP = {
    NodeColor.OBSERVED: "#4C9AFF",
    NodeColor.PARAMETER: "#FFAB4C",
    NodeColor.LATENT: "#8ED18E",
    NodeColor.NONLOCAL: "#E64980",  # отдельный цвет -- подчёркиваем нелокальность давления
}

OP_SYMBOL = {
    "+": "+", "-": "−", "*": "×",
    "ddx": "∂x", "ddy": "∂y", "laplacian": "Δ",
    "pressure_grad_x": "∇p_x", "pressure_grad_y": "∇p_y",
}


def compute_depths(tree):
    depth = {}

    def visit(node):
        if node.id in depth:
            return depth[node.id]
        depth[node.id] = 0 if not node.parents else 1 + max(visit(p) for p in node.parents)
        return depth[node.id]

    for n in tree.nodes:
        visit(n)
    return depth


def layout_layered(tree):
    depth = compute_depths(tree)
    layers = {}
    for n in tree.nodes:
        layers.setdefault(depth[n.id], []).append(n)
    pos = {}
    for d, nodes_in_layer in layers.items():
        k = len(nodes_in_layer)
        for i, n in enumerate(nodes_in_layer):
            y = (i - (k - 1) / 2.0) * 1.3
            pos[n.id] = (d * 2.0, y)
    return pos


def node_label(n):
    if n.color in (NodeColor.LATENT, NodeColor.NONLOCAL):
        return OP_SYMBOL.get(n.op, n.op)
    if n.color == NodeColor.PARAMETER:
        if "minus_one" in n.name:
            return "-1"
        return f"{n.name}\n={n.value:g}"
    return n.name


def draw_tree_graph():
    built = build_navier_stokes_tree()
    tree: PDEExpressionTree = built["tree"]
    pos = layout_layered(tree)

    fig, ax = plt.subplots(figsize=(15, 9))

    for n in tree.nodes:
        for p in n.parents:
            x1, y1 = pos[p.id]
            x2, y2 = pos[n.id]
            style = "--" if n.color == NodeColor.NONLOCAL else "-"
            arrow = FancyArrowPatch(
                (x1 + 0.3, y1), (x2 - 0.3, y2), arrowstyle="-|>", mutation_scale=10,
                color="#B5179E" if n.color == NodeColor.NONLOCAL else "#666666",
                linewidth=1.1, linestyle=style, zorder=1,
            )
            ax.add_patch(arrow)

    for n in tree.nodes:
        x, y = pos[n.id]
        radius = 0.38 if n.color in (NodeColor.LATENT, NodeColor.NONLOCAL) else 0.42
        circle = Circle((x, y), radius=radius, facecolor=COLOR_MAP[n.color],
                         edgecolor="#333333", linewidth=1.0, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, node_label(n), ha="center", va="center", fontsize=7.5, zorder=3)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "2D несжимаемые уравнения Навье-Стокса: DAG суперпозиции\n"
        "(розовым -- нелокальные вершины давления, решаются по всему полю через проекцию Чорина)",
        fontsize=12,
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP[NodeColor.OBSERVED],
               markersize=14, label="наблюдаемое поле (vx, vy)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP[NodeColor.PARAMETER],
               markersize=14, label="параметр (nu, константы)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP[NodeColor.LATENT],
               markersize=14, label="локальная операция (поточечная/дифф. оператор)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP[NodeColor.NONLOCAL],
               markersize=14, label="нелокальная операция (давление, решение по всему полю)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=2, fontsize=9, frameon=False)

    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "navier_stokes_tree_graph.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Граф сохранён: {out_path}")


def vorticity(solver, vx, vy):
    return solver.ddx(vy) - solver.ddy(vx)


def draw_physics_snapshots():
    solver = NavierStokes2D(n=96)
    vx0, vy0 = make_shear_layer_ic(n=96, delta=0.05, amplitude=0.02, seed=1)

    n_steps_total = 600
    save_every = 100
    traj_x, traj_y = solver.simulate(vx0, vy0, nu=0.001, dt=0.005,
                                       n_steps=n_steps_total, save_every=save_every)

    n_snapshots = traj_x.shape[0]
    fig, axes = plt.subplots(1, n_snapshots, figsize=(3 * n_snapshots, 3.2))
    for i in range(n_snapshots):
        w = vorticity(solver, traj_x[i], traj_y[i])
        im = axes[i].imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-w.std() * 3, vmax=w.std() * 3)
        axes[i].set_title(f"шаг {i * save_every}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    fig.suptitle("Завихренность: развитие неустойчивости Кельвина-Гельмгольца (nu=0.001)", fontsize=12)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "navier_stokes_vorticity_evolution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Картина течения сохранена: {out_path}")


if __name__ == "__main__":
    draw_tree_graph()
    draw_physics_snapshots()
