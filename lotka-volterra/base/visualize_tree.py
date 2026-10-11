"""
Визуализация DAG-структуры (дерева суперпозиции), построенного в
lotka_volterra_tree.py: три цвета вершин (наблюдаемая / параметр / операция),
рёбра -- по матрице смежности Z (j -> i, если j является входом операции i).

Рисуем два варианта:
  1) общий граф всей системы (dx/dt и dy/dt вместе, с переиспользуемым узлом x*y);
  2) два отдельных поддерева-корня (dx/dt и dy/dt) side-by-side для наглядности,
     с явным указанием, какой узел -- общий (shared subtree).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

from lotka_volterra_tree import build_lotka_volterra, ExpressionTree, NodeColor

COLOR_MAP = {
    NodeColor.OBSERVED: "#4C9AFF",     # синий -- наблюдаемая переменная
    NodeColor.PARAMETER: "#FFAB4C",    # оранжевый -- параметр
    NodeColor.LATENT: "#8ED18E",       # зелёный -- латентная операция
}

OP_SYMBOL = {"+": "+", "-": "\u2212", "*": "\u00d7", "/": "\u00f7"}


def compute_depths(tree: ExpressionTree) -> dict:
    """Глубина узла = 0 для листьев, иначе 1 + max(глубина родителей). Нужна для layered-раскладки."""
    depth = {}

    def visit(node):
        if node.id in depth:
            return depth[node.id]
        if not node.parents:
            depth[node.id] = 0
        else:
            depth[node.id] = 1 + max(visit(p) for p in node.parents)
        return depth[node.id]

    for n in tree.nodes:
        visit(n)
    return depth


def layout_layered(tree: ExpressionTree) -> dict:
    """Простая layered-раскладка без graphviz: x = глубина, y = позиция внутри слоя."""
    depth = compute_depths(tree)
    layers = {}
    for n in tree.nodes:
        layers.setdefault(depth[n.id], []).append(n)

    pos = {}
    for d, nodes_in_layer in layers.items():
        k = len(nodes_in_layer)
        for i, n in enumerate(nodes_in_layer):
            y = (i - (k - 1) / 2.0) * 1.4
            pos[n.id] = (d * 2.2, y)
    return pos


def node_label(n) -> str:
    def expr_text(node):
        if node.color in (NodeColor.OBSERVED, NodeColor.PARAMETER):
            if node.color == NodeColor.PARAMETER:
                return f"{node.name}={node.value:g}"
            return node.name

        op = OP_SYMBOL.get(node.op, node.op)
        if not node.parents:
            return op

        parts = [expr_text(parent) for parent in node.parents]
        if len(parts) == 1:
            return f"{op} {parts[0]}"
        return f"({parts[0]} {op} {parts[1]})" if len(parts) == 2 else f"{op}({', '.join(parts)})"

    text = expr_text(n)
    return text.replace(" ", "\n") if len(text) > 14 else text


def draw_tree(ax, tree: ExpressionTree, pos: dict, highlight_shared: set = None, title: str = ""):
    highlight_shared = highlight_shared or set()

    # рёбра: от родителя (входа) к текущему узлу -- по направлению потока вычисления
    for n in tree.nodes:
        for p in n.parents:
            x1, y1 = pos[p.id]
            x2, y2 = pos[n.id]
            arrow = FancyArrowPatch(
                (x1 + 0.35, y1), (x2 - 0.35, y2),
                arrowstyle="-|>", mutation_scale=12,
                color="#666666", linewidth=1.2, zorder=1,
            )
            ax.add_patch(arrow)

    # вершины
    for n in tree.nodes:
        x, y = pos[n.id]
        is_shared = n.id in highlight_shared
        edge_color = "#D6336C" if is_shared else "#333333"
        edge_width = 2.5 if is_shared else 1.0
        radius = 0.42 if n.color == NodeColor.LATENT else 0.5
        circle = Circle(
            (x, y), radius=radius, facecolor=COLOR_MAP[n.color],
            edgecolor=edge_color, linewidth=edge_width, zorder=2,
        )
        ax.add_patch(circle)
        ax.text(x, y, node_label(n), ha="center", va="center", fontsize=9,
                 fontweight="bold" if n.color == NodeColor.LATENT else "normal", zorder=3)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12)


def main():
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    built = build_lotka_volterra()
    tree: ExpressionTree = built["tree"]
    pos = layout_layered(tree)

    # узел x*y используется и в dx/dt, и в dy/dt -- находим его id для подсветки
    xy_node = None
    for n in tree.nodes:
        if n.color == NodeColor.LATENT and n.op == "*" and \
           {p.name for p in n.parents if p.color == NodeColor.OBSERVED} == {"x", "y"}:
            xy_node = n
            break
    shared_ids = {xy_node.id} if xy_node else set()

    fig, ax = plt.subplots(figsize=(11, 7))
    draw_tree(
        ax, tree, pos, highlight_shared=shared_ids,
        title="Volterra-Lotka: общий DAG суперпозиции\n"
              "(узел x·y, обведённый розовым, общий для dx/dt и dy/dt)",
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP[NodeColor.OBSERVED],
               markersize=14, label="наблюдаемая переменная (X)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP[NodeColor.PARAMETER],
               markersize=14, label="параметр (theta)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MAP[NodeColor.LATENT],
               markersize=14, label="латентная операция (V)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="#D6336C", markeredgewidth=2.5, markersize=14,
               label="переиспользуемый узел (shared subtree)"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=9, frameon=False)

    # подписи корней уравнений
    dxdt_pos = pos[built["dxdt_root"].id]
    dydt_pos = pos[built["dydt_root"].id]
    ax.annotate("= dx/dt", dxdt_pos, xytext=(dxdt_pos[0] + 0.9, dxdt_pos[1]),
                fontsize=10, fontweight="bold", va="center")
    ax.annotate("= dy/dt", dydt_pos, xytext=(dydt_pos[0] + 0.9, dydt_pos[1]),
                fontsize=10, fontweight="bold", va="center")

    plt.tight_layout()
    out_path = output_dir / "lotka_volterra_tree_graph.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Граф сохранён: {out_path}")

    # -----------------------------------------------------------------
    # Дополнительно: матрица смежности Z как heatmap -- удобно сверять
    # с картинкой графа выше и с будущей релаксированной P(G)
    # -----------------------------------------------------------------
    Z = tree.adjacency_matrix()
    labels = [node_label(n).replace("\n", " ") for n in tree.nodes]

    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im = ax2.imshow(Z, cmap="Greens", vmin=0, vmax=1)
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=90, fontsize=8)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel("j (вход)")
    ax2.set_ylabel("i (узел)")
    ax2.set_title("Матрица смежности Z: Z[i,j]=1, если j -- вход узла i")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path2 = output_dir / "lotka_volterra_adjacency_matrix.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Матрица смежности сохранена: {out_path2}")


if __name__ == "__main__":
    main()
