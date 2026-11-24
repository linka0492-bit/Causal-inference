"""Визуализации ЛР № 3. Все функции возвращают matplotlib Figure (и сохраняют её, если задан path)."""
from __future__ import annotations
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from .expr import Model
from .analysis import simulate_do_love, expanded_str, matches_lv

C_OBS, C_PAR, C_OP, C_OUT = "#4C8BF5", "#FFA94D", "#8FD18F", "#D6336C"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110})


def _save(fig, path):
    if path:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------------
# Рисунок графа (DAG)
# ----------------------------------------------------------------------------
def dag_layout(g):
    nodes, edges = g["nodes"], g["edges"]
    layer = {n["id"]: n["layer"] for n in nodes}
    ins = {n["id"]: [s for s, d in edges if d == n["id"]] for n in nodes}
    pos = {}
    leaves = [n for n in nodes if n["kind"] != "op"]
    order = {"param": 0, "obs": 1}
    leaves.sort(key=lambda n: (order[n["kind"]], n["label"]))
    for i, n in enumerate(leaves):
        pos[n["id"]] = (0.0, float(i))
    maxl = max(layer.values())
    for L in range(1, maxl + 1):
        grp = [n for n in nodes if n["kind"] == "op" and layer[n["id"]] == L]
        ys = {n["id"]: np.mean([pos[s][1] for s in ins[n["id"]]]) for n in grp}
        grp.sort(key=lambda n: ys[n["id"]])
        last = -1e9
        for n in grp:
            y = max(ys[n["id"]], last + 0.9)
            pos[n["id"]] = (float(L), y); last = y
    return pos


def draw_dag(ax, model: Model, title=None, fontsize=6):
    g = model.to_dag()
    pos = dag_layout(g)
    outs = g["outputs"]
    for s, d in g["edges"]:
        (x1, y1), (x2, y2) = pos[s], pos[d]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#666", shrinkA=11, shrinkB=11))
    col = {"obs": C_OBS, "param": C_PAR, "op": C_OP}
    for n in g["nodes"]:
        x, y = pos[n["id"]]
        ec = C_OUT if n["id"] in outs else "#333"
        ax.scatter([x], [y], s=620, c=col[n["kind"]], edgecolors=ec, linewidths=2.0 if n["id"] in outs else 0.7, zorder=3)
        ax.text(x, y, n["label"], ha="center", va="center", fontsize=fontsize, zorder=4)
    order_y = sorted(range(len(outs)), key=lambda i: pos[outs[i]][1])
    for i, o in enumerate(outs):
        x, y = pos[o]
        sgn = -1 if order_y.index(i) == 0 else 1
        ax.annotate(["dx/dt", "dy/dt"][i], xy=(x, y), xytext=(x + 0.55, y + 0.4 * sgn),
                    fontsize=7, color=C_OUT, arrowprops=dict(arrowstyle="-", color=C_OUT, lw=0.6))
    ax.set_xlim(-0.6, max(p[0] for p in pos.values()) + 1.3)
    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.axis("off"); ax.grid(False)
    if title:
        ax.set_title(title, fontsize=8)
    return g


def _phase(ax, ds, info, color="C0", small=True):
    ax.plot(ds.x, ds.y, "o", ms=2.5, c="#999", alpha=0.8, zorder=1)
    ax.plot(ds.x, ds.y, "-", lw=0.4, c="#bbb", zorder=0)
    if info is not None and info.get("sim") is not None:
        s = info["sim"]
        ax.plot(s[:, 0], s[:, 1], c=color, lw=1.3, zorder=2)
    ax.set_xlabel("заяц x", fontsize=7); ax.set_ylabel("рысь y", fontsize=7)
    ax.tick_params(labelsize=6)


# ----------------------------------------------------------------------------
# 1. Цепочка графов
# ----------------------------------------------------------------------------
def chain_unique(chain):
    """Оставляет только шаги цепочки, на которых модель изменилась. Возвращает индексы и диапазоны поколений."""
    idx, ranges = [], []
    for g, sc in enumerate(chain):
        sig = (sc.key, tuple(np.round(sc.model.get_params(), 4)))
        if idx and sig == idx[-1][1]:
            ranges[-1][1] = g
            continue
        idx.append((g, sig)); ranges.append([g, g])
    return [i for i, _ in idx], ranges


def fig_chain_pages(chain, traj, ds, per_page=6, ncols=3, path_prefix=None, dedupe=True, title="Цепочка графов", only=None):
    """Страницы с последовательностью графов: граф, метрики (параметры/латентные/наблюдаемые/ошибка),
    раскрытая формула и фазовый портрет. dedupe=True – показываются только шаги, где модель менялась."""
    if only is not None:
        ids, ranges = list(only), [[g, g] for g in only]
    elif dedupe:
        ids, ranges = chain_unique(chain)
    else:
        ids, ranges = list(range(len(chain))), [[g, g] for g in range(len(chain))]
    figs = []
    for p0 in range(0, len(ids), per_page):
        chunk = list(zip(ids[p0:p0 + per_page], ranges[p0:p0 + per_page]))
        nrows = int(np.ceil(len(chunk) / ncols))
        fig = plt.figure(figsize=(5.0 * ncols, 5.4 * nrows))
        outer = GridSpec(nrows, ncols, figure=fig, hspace=0.42, wspace=0.18)
        for k, (g, rg) in enumerate(chunk):
            sc, tr = chain[g], traj[g]
            inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[k // ncols, k % ncols], height_ratios=[1.5, 1], hspace=0.3)
            a1, a2 = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            c = sc.model.graph_counts()
            rss = tr["rss"]
            gen_txt = f"поколение {rg[0]}" + (f"–{rg[1]}" if rg[1] > rg[0] else "")
            err = (f"RSS(траектория)={rss:.0f}" if np.isfinite(rss) else "траектория расходится")
            ttl = (f"{gen_txt}   J={sc.J:.2f}\nпарам.={c['n_params']}  латентных={c['n_latent']}  наблюдаемых={c['n_observed']}\n"
                   f"ошибка: 1−R²={sc.acc:.3f}, {err}")
            draw_dag(a1, sc.model, ttl)
            a1.set_title(ttl, fontsize=8)
            a1.text(0.0, -0.02, expanded_str(sc.model, 3).replace("dx/dt", "x'").replace("dy/dt", "y'").replace(";  ", "\n"),
                    transform=a1.transAxes, fontsize=6.5, va="top", color="#333")
            _phase(a2, ds, tr, color="C3" if matches_lv(sc.model) else "C0")
        fig.suptitle(f"{title} — страница {p0 // per_page + 1}   (синие – наблюдаемые, оранжевые – параметры, зелёные – операции; красная кривая – раскрытие = Лотка–Вольтерра)",
                     fontsize=10, y=1.0)
        if path_prefix:
            fig.savefig(f"{path_prefix}{p0 // per_page + 1}.png", dpi=110, bbox_inches="tight")
        figs.append(fig)
    return figs


def fig_chain_metrics(chain, traj, path=None, title="Цепочка лучших графов: зелёный фон – раскрытая модель эквивалентна Лотке–Вольтерре"):
    """Метрики лучшей модели по всем поколениям (полная цепочка из n_gen шагов)."""
    g = np.arange(len(chain))
    cnt = [sc.model.graph_counts() for sc in chain]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax[0, 0].plot(g, [sc.J for sc in chain], "-o", ms=3); ax[0, 0].set_title("целевая функция J (меньше – лучше)")
    ax[0, 1].semilogy(g, [sc.acc for sc in chain], "-o", ms=3, c="C1"); ax[0, 1].set_title("точность: 1−R² (интегральное согласование)")
    r = [t["rss"] if np.isfinite(t["rss"]) else np.nan for t in traj]
    ax[0, 2].semilogy(g, r, "-o", ms=3, c="C3"); ax[0, 2].set_title("RSS по траектории ОДУ (тыс.²)")
    ax[1, 0].plot(g, [c["n_params"] for c in cnt], "-o", ms=3, label="параметры")
    ax[1, 0].plot(g, [c["n_latent"] for c in cnt], "-s", ms=3, label="латентные (операции)")
    ax[1, 0].plot(g, [c["n_observed"] for c in cnt], "-^", ms=3, label="наблюдаемые"); ax[1, 0].legend(); ax[1, 0].set_title("размер графа")
    ax[1, 1].plot(g, [sc.complexity for sc in chain], "-o", ms=3, c="C4"); ax[1, 1].set_title("сложность C")
    ax[1, 2].plot(g, [np.log1p(sc.instab / max(sc.acc, 1e-3)) for sc in chain], "-o", ms=3, c="C5"); ax[1, 2].set_title("неустойчивость S=ln(1+Δ/acc)")
    lv = [matches_lv(sc.model) for sc in chain]
    for a in ax.ravel():
        a.set_xlabel("поколение")
        for gi, f in zip(g, lv):
            if f: a.axvspan(gi - .5, gi + .5, color="#2f9e44", alpha=0.08, lw=0)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 2. Сходимость ГА
# ----------------------------------------------------------------------------
def fig_convergence(result, path=None):
    h = result.history
    g = [r["gen"] for r in h]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(g, [r["best_J"] for r in h], label="лучшая J"); ax[0].plot(g, [r["median_J"] for r in h], label="медиана J")
    ax[0].set_ylim(min(r["best_J"] for r in h) - 0.5, max(r["best_J"] for r in h) + 3); ax[0].legend(); ax[0].set_title("сходимость целевой функции")
    ax[1].plot(g, [r["n_unique"] for r in h], label="уникальных структур в популяции")
    ax[1].plot(g, [r["n_archive"] for r in h], label="размер архива (выборка графов)"); ax[1].legend(); ax[1].set_title("разнообразие и накопление выборки")
    ax[2].plot(g, [r["n_eval"] for r in h]); ax[2].set_title("накопленное число оценённых моделей")
    for a in ax: a.set_xlabel("поколение")
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 3. Парето: сложность – точность – устойчивость (вся выборка графов)
# ----------------------------------------------------------------------------
def pareto_mask(C, A):
    C, A = np.asarray(C), np.asarray(A)
    m = np.ones(len(C), bool)
    for i in range(len(C)):
        m[i] = not np.any((C <= C[i]) & (A <= A[i]) & ((C < C[i]) | (A < A[i])))
    return m


def fig_pareto(result, true_sc=None, path=None, lv_flags=None):
    arch = list(result.archive.values())
    C = np.array([s.complexity for s in arch]); A = np.array([s.acc for s in arch])
    S = np.array([np.log1p(s.instab / max(s.acc, 1e-3)) for s in arch])
    ok = A < 1.0
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    sc_ = ax[0].scatter(C[ok], A[ok], c=np.clip(S[ok], 0, 1.5), cmap="viridis_r", s=14, alpha=0.7)
    plt.colorbar(sc_, ax=ax[0], label="S = ln(1+Δ/acc) (клип. 1.5)")
    pm = pareto_mask(C[ok], A[ok]); Co, Ao = C[ok][pm], A[ok][pm]; o = np.argsort(Co)
    ax[0].step(Co[o], Ao[o], "r-", where="post", lw=1.3, label="фронт Парето (точность–сложность)")
    b = result.best
    ax[0].scatter([b.complexity], [b.acc], marker="*", s=300, c="gold", edgecolors="k", zorder=5, label="выбранный граф (мин. J)")
    if true_sc is not None:
        ax[0].scatter([true_sc.complexity], [true_sc.acc], marker="X", s=130, c="k", zorder=5, label="эталон ЛВ (разв. вид)")
    ax[0].set_yscale("log"); ax[0].set_xlabel("сложность C"); ax[0].set_ylabel("ошибка 1−R²"); ax[0].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"Выборка графов (n={len(arch)} уникальных структур)")
    ok2 = A < 0.7   # для более неточных моделей устойчивость не считалась
    sc2 = ax[1].scatter(A[ok2], S[ok2], c=C[ok2], cmap="plasma", s=14, alpha=0.7)
    plt.colorbar(sc2, ax=ax[1], label="сложность C")
    ax[1].scatter([b.acc], [np.log1p(b.instab / max(b.acc, 1e-3))], marker="*", s=300, c="gold", edgecolors="k", zorder=5)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("ошибка 1−R²"); ax[1].set_ylabel("неустойчивость S"); ax[1].set_title("Точность против неустойчивости (модели с 1−R² < 0.7)")
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 4. Выборка и аппроксимация выбранным графом
# ----------------------------------------------------------------------------
def fig_fit(info, model, ds, title="Выбранный граф: данные и аппроксимация", path=None):
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.4, 1])
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    a3 = fig.add_subplot(gs[0, 2]); a4 = fig.add_subplot(gs[1, :2]); a5 = fig.add_subplot(gs[1, 2])
    t, s = info["t"], info["sim"]
    a1.plot(t, s[:, 0], c="C0", label="модель"); a1.plot(ds.t, ds.x, "ko", ms=3, label="данные"); a1.set_title("заяц x(t)"); a1.legend(); a1.set_xlabel("год")
    a2.plot(t, s[:, 1], c="C1", label="модель"); a2.plot(ds.t, ds.y, "ko", ms=3, label="данные"); a2.set_title("рысь y(t)"); a2.legend(); a2.set_xlabel("год")
    a3.plot(ds.x, ds.y, "ko", ms=3); a3.plot(s[:, 0], s[:, 1], c="C3"); a3.set_title("фазовый портрет"); a3.set_xlabel("x"); a3.set_ylabel("y")
    # невязки
    from .fit import simulate
    sim_pts = simulate(info["model"], info["x0"], info["y0"], ds.t)
    rx, ry = ds.x - sim_pts[:, 0], ds.y - sim_pts[:, 1]
    a4.bar(ds.t - 0.2, rx, width=0.4, label="x", color="C0"); a4.bar(ds.t + 0.2, ry, width=0.4, label="y", color="C1")
    a4.axhline(0, c="k", lw=0.6); a4.set_title(f"невязки (RSS = {info['rss']:.1f})"); a4.legend(); a4.set_xlabel("год")
    a5.axis("off"); a5.grid(False)
    import textwrap
    txt = ("Граф:\n" + "\n".join(textwrap.wrap(str(model).replace(";  ", "\n").replace("\n", " ;  "), 52)) + "\n\nРаскрыто:\n" + expanded_str(model, 4).replace(";  ", "\n"))
    a5.text(0, 1, txt, va="top", fontsize=8, wrap=True, family="DejaVu Sans")
    a5.set_title("формулы", fontsize=9)
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 5. Разложение целевой функции и устойчивость выбора к весам
# ----------------------------------------------------------------------------
def fig_objective_terms(result, topk=12, path=None, arch=None):
    from .objective import objective_terms
    w = result.config.weights
    top = (arch if arch is not None else sorted(result.archive.values(), key=lambda s: s.J))[:topk]
    t = [objective_terms(s, w) for s in top]
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(top))
    # знак: точность отрицательна (ln acc<0); показываем три слагаемых рядом
    ax.bar(x - 0.27, [a["accuracy"] for a in t], 0.27, label="w·ln(acc) (точность)", color="C0")
    ax.bar(x, [a["complexity"] for a in t], 0.27, label="w·C (сложность)", color="C1")
    ax.bar(x + 0.27, [a["stability"] for a in t], 0.27, label="w·S (неустойчивость)", color="C2")
    ax.plot(x, [s.J for s in top], "k-o", ms=4, label="J (сумма)")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([f"#{i + 1}\n" + expanded_str(s_.model, 2).replace("dx/dt", "x'").replace("dy/dt", "y'").replace(";  ", "\n").replace("*", "")[:46] for i, s_ in zip(x, top)], fontsize=5.5)
    ax.set_title(f"Вклад трёх критериев в J для {topk} лучших классов эквивалентных графов (подписи – раскрытая формула)"); ax.legend(ncol=4, fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def fig_weight_sensitivity(result, true_sc=None, path=None, n=25, arch=None):
    """Как меняется выбранный граф при изменении весов сложности и устойчивости (пересчёт J по архиву)."""
    arch = list(arch if arch is not None else result.archive.values())
    A = np.array([max(s.acc, 1e-3) for s in arch]); C = np.array([s.complexity for s in arch])
    S = np.array([np.log1p(s.instab / max(s.acc, 1e-3)) for s in arch]); fin = np.isfinite(A) & np.isfinite(S) & (A < 1)
    arch = [s for s, f in zip(arch, fin) if f]; A, C, S = A[fin], C[fin], S[fin]
    wc = np.geomspace(0.002, 0.5, n); ws = np.concatenate([[0], np.geomspace(0.05, 20, n - 1)])
    pick = np.zeros((n, n), int)
    for i, c in enumerate(wc):
        for j, s in enumerate(ws):
            pick[i, j] = int(np.argmin(np.log(A) + c * C + s * S))
    share = {u: np.mean(pick == u) for u in set(pick.ravel())}
    uniq = sorted(share, key=lambda u: -share[u])
    remap = {u: k for k, u in enumerate(uniq)}
    img = np.vectorize(remap.get)(pick)
    cmap = plt.get_cmap("tab20", max(len(uniq), 2))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.05, 1]})
    ax[0].imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=-0.5, vmax=max(len(uniq), 2) - 0.5)
    ax[0].set_xticks(np.arange(0, n, 4)); ax[0].set_xticklabels([f"{ws[k]:.2g}" for k in range(0, n, 4)])
    ax[0].set_yticks(np.arange(0, n, 4)); ax[0].set_yticklabels([f"{wc[k]:.3f}" for k in range(0, n, 4)])
    ax[0].set_xlabel("вес устойчивости w_stab"); ax[0].set_ylabel("вес сложности w_cmp (на вершину)")
    cur = result.config.weights
    ax[0].plot(np.interp(cur.w_stab, ws, np.arange(n)), np.interp(cur.w_cmp, wc, np.arange(n)), "k*", ms=16, mfc="gold")
    ax[0].set_title("Какой граф оптимален при разных весах J (цвет = номер графа; ★ – использованные веса)")
    ax[0].grid(False)
    ax[1].axis("off"); ax[1].grid(False)
    y0 = 1.0
    for k, u in enumerate(uniq[:9]):
        sc_ = arch[u]
        ex = expanded_str(sc_.model, 3).replace("dx/dt", "x'").replace("dy/dt", "y'").replace(";  ", "\n      ")
        ax[1].add_patch(plt.Rectangle((0, y0 - 0.085), 0.03, 0.06, color=cmap(k), transform=ax[1].transAxes, clip_on=False))
        ax[1].text(0.045, y0 - 0.02, f"[{k}] {share[u] * 100:.1f}% площади · C={sc_.complexity:.1f} · 1−R²={sc_.acc:.3f} · S={np.log1p(sc_.instab / max(sc_.acc, 1e-3)):.2f}\n      {ex}",
                   va="top", fontsize=7.2, transform=ax[1].transAxes)
        y0 -= 0.108
    ax[1].set_title("Графы, которые становятся оптимальными (по убыванию площади)", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 6. Структурная интервенция do(u=1): найденная модель против эталона и конкурентов
# ----------------------------------------------------------------------------
def fig_do_love(models: dict, ds, lam_fracs=(0.25, 0.5, 0.75), beta=0.0248, t_do=1910.0, path=None, tmax=1940):
    """models: name -> (Model, x0, y0, style). Фазовые портреты после вмешательства."""
    fig, ax = plt.subplots(1, len(lam_fracs) + 1, figsize=(5 * (len(lam_fracs) + 1), 5))
    t = np.linspace(ds.t[0], tmax, 800)
    for name, (m, x0, y0, st) in models.items():
        tt, s0 = simulate_do_love(m, x0, y0, t, 0.0, t_do)
        if s0 is not None:
            ax[0].plot(s0[:, 0], s0[:, 1], **st, label=name)
    ax[0].set_title("без интервенции (lam=0), t ≥ 1910")
    for a, f in zip(ax[1:], lam_fracs):
        lam = f * beta
        for name, (m, x0, y0, st) in models.items():
            tt, s = simulate_do_love(m, x0, y0, t, lam, t_do)
            if s is None:
                continue
            a.plot(s[:, 0], s[:, 1], **st, label=name)
        a.set_title(f"do(u=1), lam = {f}·β = {lam:.4f}")
    for a in ax:
        a.set_xlabel("заяц x (тыс.)"); a.set_ylabel("рысь y (тыс.)"); a.set_ylim(0, 450); a.set_xlim(0, 110)
    ax[0].legend(fontsize=7)
    fig.suptitle("Проверка причинно-следственной структуры: интервенция «взаимная любовь» (добавляется lam·u·x·y в оба уравнения)", y=1.0)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 7. Устойчивость найденной модели к интервенциям в данные (W1/σ, KS D) – как в ЛР № 2
# ----------------------------------------------------------------------------
def fig_intervention_heatmaps(tab, param_labels, path=None):
    names = [k for k in tab["W1"] if k != "__method_noise__"]
    W = np.array([tab["W1"][k] for k in names]); D = np.array([tab["D"][k] for k in names])
    P = np.array([tab["p"][k] for k in names])
    nW, nD = tab["W1"]["__method_noise__"], tab["D"]["__method_noise__"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 0.9 * len(names) + 2.8))
    for a, M, ttl, noise in [(ax[0], W, "W1 / σ(base)", nW), (ax[1], D, "статистика D (Колмогоров–Смирнов)", nD)]:
        im = a.imshow(M, cmap="viridis", aspect="auto")
        a.set_xticks(range(len(param_labels))); a.set_xticklabels(param_labels, rotation=30)
        a.set_yticks(range(len(names))); a.set_yticklabels(names)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                star = "*" if P[i, j] < 0.05 / M.size else ""
                a.text(j, i, f"{M[i, j]:.2f}{star}", ha="center", va="center", color="w" if M[i, j] < M.max() * 0.6 else "k", fontsize=8)
        plt.colorbar(im, ax=a, fraction=0.04)
        a.set_title(f"{ttl}\nшум метода (base vs base): среднее {np.mean(noise):.2f}"); a.grid(False)
    fig.suptitle("Сдвиг распределения параметров выбранного графа при интервенциях в данные (* – значимо, КС, Бонферрони)", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


def fig_param_hists(tab, scen_samples, param_labels, path=None):
    """Гистограммы параметров по мини-батчам: base и выбранные интервенции."""
    base = tab["base_samples"]
    n = base.shape[1]
    fig, ax = plt.subplots(1, n, figsize=(3.4 * n, 3.4))
    ax = np.atleast_1d(ax)
    for j in range(n):
        ax[j].hist(base[:, j], bins=25, density=True, histtype="step", lw=2, color="k", label="base")
        for name, smp in scen_samples.items():
            ax[j].hist(smp[:, j], bins=25, density=True, histtype="step", lw=1.2, label=name)
        ax[j].set_title(param_labels[j]); ax[j].set_xlabel("значение параметра")
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------------------------
# 8. Матрица инцидентности A выбранного графа
# ----------------------------------------------------------------------------
def fig_incidence(model: Model, path=None):
    A, labels = model.incidence_matrix()
    g = model.to_dag()
    names = [f"{n['id']}:{n['label']}" for n in g["nodes"]]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.1]})
    draw_dag(ax[0], model, "граф выбранной модели", fontsize=8)
    ax[1].imshow(A, cmap="Greys", vmin=0, vmax=1)
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(names, rotation=90, fontsize=7)
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=7)
    ax[1].set_xlabel("куда (потребитель)"); ax[1].set_ylabel("откуда (вход)")
    ax[1].set_title("Бинарная матрица инцидентности A (A[i,j]=1: ребро i→j)")
    ax[1].grid(False)
    fig.tight_layout()
    return _save(fig, path)
