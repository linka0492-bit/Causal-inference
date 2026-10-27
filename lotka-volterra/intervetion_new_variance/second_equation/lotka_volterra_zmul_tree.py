"""
DAG суперпозиции для Лотки-Вольтерры с мультипликативной переменной z:

    dx/dt = (alpha - beta*y) * x * z
    dy/dt = (-gamma + delta*x) * y * z

Использует ExpressionTree из lotka_volterra_tree.py без изменений.
При z = 1 получается исходная модель (z масштабирует время: dt -> z*dt).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lotka_volterra_tree import ExpressionTree, NodeColor
from lotka_volterra_z_tree import evaluator


def build_lotka_volterra_zmul() -> dict:
    tree = ExpressionTree()
    x = tree.observed("x")
    y = tree.observed("y")
    z = tree.observed("z")                      # НОВАЯ переменная

    alpha = tree.parameter("alpha", 0.481)
    beta = tree.parameter("beta", 0.0248)
    gamma = tree.parameter("gamma", 0.926)
    delta = tree.parameter("delta", 0.0275)

    # dx/dt = (alpha - beta*y) * x * z
    by = tree.op("*", beta, y)
    a_by = tree.op("-", alpha, by)
    t1 = tree.op("*", a_by, x)
    dxdt_root = tree.op("*", t1, z)             # НОВЫЙ корень

    # dy/dt = (delta*x - gamma) * y * z
    dx_ = tree.op("*", delta, x)
    d_g = tree.op("-", dx_, gamma)
    t2 = tree.op("*", d_g, y)
    dydt_root = tree.op("*", t2, z)             # НОВЫЙ корень

    return {
        "tree": tree, "x": x, "y": y, "z": z,
        "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        "dxdt_root": dxdt_root, "dydt_root": dydt_root,
        "new_ids": {z.id, dxdt_root.id, dydt_root.id},
        "roots": [dxdt_root, dydt_root],
    }


def compile_tree(built=None):
    """Python-функция f(x,y,z,alpha,beta,gamma,delta)->(dx,dy), сгенерированная из DAG."""
    b = built or build_lotka_volterra_zmul()
    lines = []
    for n in b["tree"].nodes:
        if not n.parents:
            lines.append(f"    v{n.id} = {n.name}")
        else:
            a = [f"v{p.id}" for p in n.parents]
            if n.op == "+":
                e = " + ".join(a)
            elif n.op == "-":
                e = a[0] + "".join(f" - {t}" for t in a[1:])
            elif n.op == "*":
                e = " * ".join(a)
            else:
                e = f"{a[0]} / {a[1]}"
            lines.append(f"    v{n.id} = {e}")
    src = ("def f(x, y, z, alpha, beta, gamma, delta):\n" + "\n".join(lines) +
           f"\n    return v{b['dxdt_root'].id}, v{b['dydt_root'].id}\n")
    ns = {}
    exec(src, ns)
    return ns["f"]


def draw_tree(built: dict, path: str):
    tree, roots, new_ids = built["tree"], built["roots"], built["new_ids"]
    layer = {}
    for n in tree.nodes:
        layer[n.id] = 0 if not n.parents else 1 + max(layer[p.id] for p in n.parents)
    L = max(layer.values())

    leaf_order = ["z", "x", "y", "alpha", "beta", "gamma", "delta"]
    leaves = {n.name: n for n in tree.nodes if not n.parents}
    pos = {leaves[nm].id: (0, -i) for i, nm in enumerate(leaf_order)}
    for l in range(1, L + 1):
        ops = [n for n in tree.nodes if n.parents and layer[n.id] == l]
        for n in ops:
            pos[n.id] = (l, np.mean([pos[p.id][1] for p in n.parents]))
        ys = sorted(ops, key=lambda n: -pos[n.id][1])
        for a, b in zip(ys, ys[1:]):
            if pos[a.id][1] - pos[b.id][1] < 0.9:
                pos[b.id] = (l, pos[a.id][1] - 0.9)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for n in tree.nodes:
        for p in n.parents:
            (x0, y0), (x1, y1) = pos[p.id], pos[n.id]
            new_edge = n.id in new_ids and p.id in new_ids
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#c2185b" if new_edge else "#777",
                                        lw=1.6 if new_edge else 0.9, shrinkA=14, shrinkB=14,
                                        connectionstyle=f"arc3,rad={0.12 if (layer[n.id] - layer[p.id]) > 1 else 0}"))
    cmap = {NodeColor.OBSERVED: "#4d94ff", NodeColor.PARAMETER: "#ffaa4d", NodeColor.LATENT: "#8fce8f"}
    for n in tree.nodes:
        px, py = pos[n.id]
        new = n.id in new_ids
        ax.scatter([px], [py], s=1500, c=cmap[n.color],
                   edgecolors="#d81b60" if new else "#333", linewidths=3 if new else 1, zorder=3)
        lab = n.op if n.color == NodeColor.LATENT else (
            f"{n.name}\n={n.value:g}" if n.color == NodeColor.PARAMETER else n.name)
        ax.text(px, py, lab, ha="center", va="center", fontsize=9, zorder=4)
    for r, name in zip(roots, ["dx/dt", "dy/dt"]):
        px, py = pos[r.id]
        ax.text(px + 0.35, py, name, va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.6, L + 1.0)
    ax.set_title("DAG: Лотка–Вольтерра с мультипликативной z. Розовым обведено новое "
                 "(z и два корня «·z»)", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def demo():
    built = build_lotka_volterra_zmul()
    f = compile_tree(built)
    fx, fy = evaluator(built["dxdt_root"]), evaluator(built["dydt_root"])
    rng = np.random.default_rng(0)
    for _ in range(1000):
        e = {k: rng.uniform(0.1, 3) for k in ["x", "y", "z", "alpha", "beta", "gamma", "delta"]}
        rx = (e["alpha"] - e["beta"] * e["y"]) * e["x"] * e["z"]
        ry = (-e["gamma"] + e["delta"] * e["x"]) * e["y"] * e["z"]
        assert np.isclose(fx(e), rx) and np.isclose(fy(e), ry)
        assert np.allclose(f(e["x"], e["y"], e["z"], e["alpha"], e["beta"], e["gamma"], e["delta"]), (rx, ry))
        e["z"] = 1.0
        assert np.isclose(fx(e), (e["alpha"] * e["x"] - e["beta"] * e["x"] * e["y"]))
        assert np.isclose(fy(e), (e["delta"] * e["x"] * e["y"] - e["gamma"] * e["y"]))
    print("Проверка на 1000 случайных точках: дерево = формулам из задания; при z=1 -> исходная модель.")
    print("Размер матрицы смежности:", built["tree"].adjacency_matrix().shape,
          ", рёбер:", int(built["tree"].adjacency_matrix().sum()))
    draw_tree(built, "lv_zmul_tree.png")
    print("Рисунок: lv_zmul_tree.png")


if __name__ == "__main__":
    demo()
