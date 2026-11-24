"""Графики: коллекция, граф модели (DAG), последовательность прогнозируемых графов,
аппроксимация выборки, сравнение методов, настройка гиперпараметров."""
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from .library import term_name
from .simulate import polish_params, simulate

BLUE, ORANGE, GREEN, ROOT = "#4a8cf7", "#ffa64d", "#8fd18f", "#d6336c"


# ---------------------------------------------------------------- граф модели
def build_dag(sample, Z, C):
    """Узлы и рёбра графа суперпозиции: наблюдаемые (синие), параметры (оранжевые),
    операции (зелёные), корни dx/dt (с малиновой обводкой). Одинаковые мономы рисуются один раз."""
    terms, names = sample.terms, sample.names
    nodes, edges = {}, []
    for v in range(sample.n_vars):
        nodes[("var", v)] = dict(col=0, label=names[v], kind="var")
    for i in range(sample.n_eq):
        if Z[i].any():
            nodes[("sum", i)] = dict(col=3, label="+", kind="root", name=f"d{names[i]}/dt")
    for i, j in np.argwhere(Z):
        e = terms[j]
        deg = sum(e)
        src = None
        if deg >= 2:
            key = ("mon", e)
            if key not in nodes:
                nodes[key] = dict(col=1, label="×", kind="op")
                for v, p in enumerate(e):
                    if p:
                        edges.append((("var", v), key))
                        if p > 1:
                            edges.append((("var", v), key))      # вторая стрелка — степень монома
            src = key
        elif deg == 1:
            src = ("var", int(np.argmax(e)))
        pk = ("par", int(i), int(j))
        nodes[pk] = dict(col=0, label=f"{C[i, j]:.3g}", kind="par")
        if src is None:
            edges.append((pk, ("sum", int(i))))
        else:
            ck = ("cx", int(i), int(j))
            nodes[ck] = dict(col=2, label="×", kind="op")
            edges += [(pk, ck), (src, ck), (ck, ("sum", int(i)))]
    return nodes, edges


def draw_model_graph(ax, sample, Z, C, title="", max_edges=12):
    ax.set_axis_off()
    if Z.sum() > max_edges:
        ax.text(0.5, 0.5, f"{int(Z.sum())} рёбер\n(граф слишком плотный\nдля рисунка)", ha="center", va="center", fontsize=8)
        ax.set_title(title, fontsize=8)
        return
    nodes, edges = build_dag(sample, Z, C)
    cols = {}
    for k, nd in nodes.items():
        cols.setdefault(nd["col"], []).append(k)
    pos = {}
    for c, ks in cols.items():
        ks = sorted(ks, key=lambda k: (0 if k[0] == "var" else 1, k[1:]))
        n = len(ks)
        for r, k in enumerate(ks):
            pos[k] = (c, 1 - (r + 0.5) / n)
    for a, b in edges:
        if a in pos and b in pos:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.45", shrinkA=6, shrinkB=6))
    colors = dict(var=BLUE, par=ORANGE, op=GREEN, root=GREEN)
    for k, nd in nodes.items():
        x, y = pos[k]
        root = nd["kind"] == "root"
        ax.scatter([x], [y], s=260, c=colors[nd["kind"]], edgecolors=ROOT if root else "k",
                   linewidths=1.6 if root else 0.4, zorder=3)
        ax.text(x, y, nd["label"], ha="center", va="center", fontsize=6, zorder=4)
        if root:
            ax.text(x + 0.2, y, nd["name"], ha="left", va="center", fontsize=7, color=ROOT)
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=8)


# ---------------------------------------------------------------- матрица смежности
def draw_adjacency(ax, sample, Z, flips=(), title=""):
    terms = sample.terms
    img = np.where(Z, 0.35, 1.0)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect="auto")
    for i, j, d in flips:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#2ca25f" if d > 0 else "#f0883e", alpha=0.8))
    for i, j in np.argwhere(sample.Z):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ROOT, lw=1.6))
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels([term_name(e, sample.names) for e in terms], rotation=90, fontsize=5)
    ax.set_yticks(range(sample.n_eq))
    ax.set_yticklabels([f"d{sample.names[i]}/dt" for i in range(sample.n_eq)], fontsize=6)
    ax.set_title(title, fontsize=7)


def plot_sequence(fc, n_show=7, fname=None):
    """Последовательность графов прогноза: сверху граф модели, снизу матрица смежности
    (серая — включённые рёбра, красная рамка — эталон, зелёное/оранжевое — добавлено/удалено на шаге)."""
    st, s = fc.steps, fc.prep.s
    ids = sorted({0, len(st) - 1, *np.linspace(0, len(st) - 1, min(n_show, len(st))).astype(int)})
    fig, axes = plt.subplots(2, len(ids), figsize=(2.7 * len(ids), 6.4), squeeze=False)
    for c, k in enumerate(ids):
        step = st[k]
        C = np.array([fc.prep.eqs[i].physical(step.W[i]) for i in range(fc.prep.n_eq)])
        kind = {"start": "старт: полный граф", "regression": "шаг", "prune": "прунинг"}[step.kind]
        draw_model_graph(axes[0, c], s, step.Z, C, f"{kind} {k}: {int(step.Z.sum())} рёбер")
        tp = (step.Z & s.Z).sum()
        f1 = 2 * tp / (step.Z.sum() + s.Z.sum())
        draw_adjacency(axes[1, c], s, step.Z, step.flips, f"F1 по рёбрам {f1:.2f}")
    fig.suptitle(f"Последовательность графов: {s.system} (красная рамка — эталон)", fontsize=10)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=130)
    return fig


def plot_sequence_metrics(fc, fname=None):
    s, st = fc.prep.s, fc.steps
    n = [int(x.Z.sum()) for x in st]
    f1 = [2 * (x.Z & s.Z).sum() / (x.Z.sum() + s.Z.sum()) for x in st]
    rss = [np.mean(x.rss) for x in st]
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.6))
    ax[0].plot(n, "o-"); ax[0].axhline(s.Z.sum(), c=ROOT, ls="--"); ax[0].set_title("число рёбер (пунктир — эталон)", fontsize=8)
    ax[1].plot(f1, "o-"); ax[1].set_ylim(0, 1.05); ax[1].set_title("F1 по рёбрам относительно эталона", fontsize=8)
    ax[2].semilogy(np.maximum(rss, 1e-12), "o-"); ax[2].set_title("необъяснённая доля (интегр. согласование)", fontsize=8)
    for a in ax:
        a.set_xlabel("шаг")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=130)
    return fig


def plot_gamma(fc, n_show=4, fname=None):
    s = fc.prep.s
    gs = [st for st in fc.steps if st.Gamma is not None][:n_show]
    if not gs:
        return None
    fig, axes = plt.subplots(1, len(gs), figsize=(3 * len(gs), 2.2), squeeze=False)
    for k, (ax, st) in enumerate(zip(axes[0], gs)):
        im = ax.imshow(st.Gamma, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i, j in np.argwhere(s.Z):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ROOT, lw=1.4))
        ax.set_xticks(range(len(s.terms)))
        ax.set_xticklabels([term_name(e, s.names) for e in s.terms], rotation=90, fontsize=5)
        ax.set_yticks(range(s.n_eq)); ax.set_yticklabels([f"d{n}/dt" for n in s.names[:s.n_eq]], fontsize=6)
        ax.set_title(f"Γ на шаге {k + 1}", fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    if fname:
        fig.savefig(fname, dpi=130, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------- аппроксимация выборки
def plot_fit(fc, polish=True, n_show=None, fname=None):
    """Выборка и её аппроксимация спрогнозированной моделью: ряды, фазовый портрет, невязки,
    граф и формулы. polish=True — параметры уточняются по траектории ОДУ (как в ЛР № 3)."""
    s, st = fc.prep.s, fc.steps[-1]
    Z, C = st.Z, fc.coef()
    if polish:
        try:
            C, x0, rss = polish_params(s, Z, C)
        except Exception:
            x0, rss = s.X[0, :s.n_eq], np.nan
    else:
        x0 = s.X[0, :s.n_eq]
    n_show = n_show or len(s.t)
    tt = np.linspace(s.t[0], s.t[n_show - 1], 400)
    Y = simulate(s, Z, C, x0, tt)
    Yd = simulate(s, Z, C, x0, s.t[:n_show])
    obs = s.X[:n_show, :s.n_eq]
    fig = plt.figure(figsize=(12, 7))
    gs = GridSpec(2, 3, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    for v in range(s.n_eq):
        ax.plot(s.t[:n_show], obs[:, v], "o", ms=3, color=f"C{v}")
        ax.plot(tt, Y[:, v], "-", color=f"C{v}", label=s.names[v])
    ax.set_title("данные (точки) и модель (линия)", fontsize=9); ax.legend(fontsize=7); ax.set_xlabel("t")
    ax = fig.add_subplot(gs[0, 1])
    if s.n_eq >= 2:
        ax.plot(obs[:, 0], obs[:, 1], "o", c="0.5", ms=3); ax.plot(Y[:, 0], Y[:, 1], "r-", lw=1)
        ax.set_xlabel(s.names[0]); ax.set_ylabel(s.names[1]); ax.set_title("фазовый портрет", fontsize=9)
    else:
        ax.plot(s.t[:n_show], obs[:, 0] - Yd[:, 0]); ax.set_title("невязка", fontsize=9)
    ax = fig.add_subplot(gs[0, 2])
    w = 0.8 / s.n_eq
    for v in range(s.n_eq):
        ax.bar(s.t[:n_show] + (v - (s.n_eq - 1) / 2) * w * np.median(np.diff(s.t)), obs[:, v] - Yd[:, v],
               width=w * np.median(np.diff(s.t)), color=f"C{v}")
    ax.set_title(f"невязки по траектории (RSS = {np.nansum((obs - Yd) ** 2):.4g})", fontsize=9)
    ax = fig.add_subplot(gs[1, :2])
    draw_model_graph(ax, s, Z, C, "граф спрогнозированной модели")
    ax = fig.add_subplot(gs[1, 2]); ax.set_axis_off()
    from .forecaster import Forecast
    lines = []
    for i in range(s.n_eq):
        parts = [f"{C[i, j]:+.4g}·{term_name(e, s.names)}" for j, e in enumerate(s.terms) if Z[i, j]]
        lines.append(f"прогноз d{s.names[i]}/dt =\n   " + " ".join(parts))
    true = []
    for i in range(s.n_eq):
        parts = [f"{s.W[i, j]:+.4g}·{term_name(e, s.names)}" if np.isfinite(s.W[i, j]) else term_name(e, s.names)
                 for j, e in enumerate(s.terms) if s.Z[i, j]]
        true.append(f"эталон d{s.names[i]}/dt =\n   " + " ".join(parts))
    ax.text(0, 1, "\n".join(lines + [""] + true), va="top", fontsize=7, family="monospace", wrap=True)
    fig.suptitle(f"{s.system} (данные: {s.tag}); структура {'совпала' if (Z == s.Z).all() else 'не совпала'} с эталоном", fontsize=10)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=130)
    return fig


# ---------------------------------------------------------------- коллекция
def plot_collection(samples, ncols=4, fname=None):
    seen, picks = set(), []
    for s in samples:
        if s.system not in seen:
            seen.add(s.system); picks.append(s)
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows), squeeze=False)
    for ax, s in zip(axes.ravel(), picks):
        for v in range(s.n_vars):
            ax.plot(s.t, s.X[:, v], ".-", ms=2, lw=0.7, label=s.names[v])
        ax.set_title(f"{s.system}: {int(s.Z.sum())} рёбер в эталоне", fontsize=8)
        ax.legend(fontsize=5)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=130)
    return fig


# ---------------------------------------------------------------- сравнение методов и настройка
def plot_tuning(rows, fname=None):
    import collections
    ridges = sorted({r["ridge"] for r in rows})
    fig, axes = plt.subplots(1, len(ridges), figsize=(4.2 * len(ridges), 3), squeeze=False)
    taus = sorted({r["tau"] for r in rows})
    combos = sorted({(r["max_flips"], r["final_prune"]) for r in rows})
    vmin, vmax = min(r["J"] for r in rows), max(r["J"] for r in rows)
    for ax, rg in zip(axes[0], ridges):
        M = np.full((len(taus), len(combos)), np.nan)
        for r in rows:
            if r["ridge"] == rg:
                M[taus.index(r["tau"]), combos.index((r["max_flips"], r["final_prune"]))] = r["J"]
        im = ax.imshow(M, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([f"k={k}\n{'прун.' if p else '—'}" for k, p in combos], fontsize=6)
        ax.set_yticks(range(len(taus))); ax.set_yticklabels([f"τ={t}" for t in taus], fontsize=7)
        ax.set_title(f"ridge = {rg}", fontsize=9)
    fig.suptitle("J на валидации (меньше — лучше)", fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    if fname:
        fig.savefig(fname, dpi=130, bbox_inches="tight")
    return fig


def plot_methods(results, metric="exact", title="", fname=None):
    """results: {метод: {группа: evaluate(...)}} или {метод: evaluate(...)}."""
    methods = list(results)
    first = results[methods[0]]
    if "J" in first:
        results = {m: {"все": r} for m, r in results.items()}
        first = results[methods[0]]
    groups = list(first)
    fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(groups) * len(methods) / 2.5), 3.3))
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        ax.bar(np.arange(len(groups)) + (k - (len(methods) - 1) / 2) * w, [results[m][g][metric] for g in groups], w, label=m)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel({"exact": "доля точно угаданных структур", "f1": "F1 по рёбрам"}.get(metric, metric))
    ax.set_ylim(0, 1.02); ax.legend(fontsize=6); ax.set_title(title, fontsize=9)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=130)
    return fig


def plot_per_system(per_sample, fname=None):
    """Доля точных совпадений по системам (по результатам evaluate()['per_sample'])."""
    by = {}
    for r in per_sample:
        by.setdefault(r["system"], []).append(r["exact"])
    names = list(by)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(names, [np.mean(by[n]) for n in names], color="C0")
    ax.set_ylim(0, 1.02); ax.set_ylabel("доля точных структур"); plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    if fname:
        fig.savefig(fname, dpi=130)
    return fig
