"""
Интервенция «взаимная любовь» в модели Volterra-Lotka и её влияние на графики.

Берём параметры, подогнанные Левенбергом-Марквардтом по данным Hudson Bay, и
в момент t_on (по умолчанию 1910 год) включаем do(u=1): член lam*x*y добавляется
к обоим уравнениям (см. build_lv_with_love). Сравниваем несколько сил lam,
выраженных как доля f от beta.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from lv_common import LVModel, fit, T_OBS, DATA, YEARS, HARE, LYNX
from visualize_tree import draw_tree, layout_layered

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

T_ON = 10.0                      # момент вмешательства, лет от 1900 (т.е. 1910)
T_END = 40.0                     # горизонт прогноза (до 1940)
FRACS = [0.0, 0.25, 0.5, 0.75]   # lam = f * beta; f=0 -- без интервенции
COLORS = ["black", "tab:green", "tab:orange", "tab:red"]


def summarize(sim, t):
    """Метрики после вмешательства: размах и средний период по пикам зайца."""
    m = t >= T_ON
    hare, lynx, tt = sim[m, 0], sim[m, 1], t[m]
    pk, _ = find_peaks(hare, prominence=5)
    period = np.diff(tt[pk]).mean() if len(pk) >= 2 else np.nan
    return hare.max(), hare.min(), lynx.max(), lynx.min(), period


def main():
    model = LVModel(rtol=1e-9, atol=1e-9)
    p_hat, rss0, _ = fit(model)
    alpha, beta, gamma, delta, x0, y0 = p_hat
    print("Базовая подгонка (LM):")
    print(f"  alpha={alpha:.4f} beta={beta:.4f} gamma={gamma:.4f} delta={delta:.4f} "
          f"x0={x0:.2f} y0={y0:.2f}  RSS={rss0:.1f}")
    print(f"\nИнтервенция включается в t={T_ON:g} (год {1900 + T_ON:g}), lam = f*beta\n")

    t = np.linspace(0, T_END, 1601)
    sims = {}
    rows = []
    for f in FRACS:
        lam = f * beta
        sim = model.simulate(p_hat, t, lam=lam, t_on=T_ON)
        sims[f] = sim
        rss = np.sum((model.simulate(p_hat, T_OBS, lam=lam, t_on=T_ON) - DATA) ** 2)
        hmax, hmin, lmax, lmin, per = summarize(sim, t)
        # равновесие нетривиальной точки после включения: x*=gamma/(delta+lam), y*=alpha/(beta-lam)
        xs, ys = gamma / (delta + lam), alpha / (beta - lam)
        rows.append((f, lam, xs, ys, hmax, hmin, lmax, lmin, per, rss))

    print("  f     lam     x*     y*   | заяц max/min | рысь max/min | период | RSS по данным")
    for f, lam, xs, ys, hmax, hmin, lmax, lmin, per, rss in rows:
        print(f"  {f:4.2f} {lam:7.4f} {xs:6.1f} {ys:6.1f} | {hmax:6.1f}/{hmin:5.1f} | "
              f"{lmax:6.1f}/{lmin:5.1f} | {per:6.2f} | {rss:9.1f}")

    # ---------------- рисунок 1: траектории и фазовый портрет ----------------
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2)
    ax_h, ax_l, ax_p = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    for f, c in zip(FRACS, COLORS):
        lab = "без интервенции" if f == 0 else f"lam = {f:g}·beta"
        ax_h.plot(1900 + t, sims[f][:, 0], color=c, label=lab, lw=1.6 if f else 2)
        ax_l.plot(1900 + t, sims[f][:, 1], color=c, label=lab, lw=1.6 if f else 2)
        m = t >= T_ON
        ax_p.plot(sims[f][m, 0], sims[f][m, 1], color=c, label=lab, lw=1.4)
    ax_h.scatter(YEARS, HARE, color="tab:blue", zorder=5, s=18, label="данные")
    ax_l.scatter(YEARS, LYNX, color="tab:blue", zorder=5, s=18, label="данные")
    for ax, name in ((ax_h, "Заяц"), (ax_l, "Рысь")):
        ax.axvline(1900 + T_ON, color="gray", ls=":", lw=1.2)
        ax.text(1900 + T_ON - 0.4, ax.get_ylim()[1] * 0.5, "do(u=1)", color="gray", fontsize=9,
                rotation=90, ha="right", va="center")
        ax.set_title(f"{name}: траектории при включении «взаимной любви»")
        ax.set_xlabel("год")
        ax.set_ylabel("численность (тыс.)")
        ax.grid(alpha=0.3)
    ax_h.legend(fontsize=8)
    ax_p.set_title("Фазовый портрет после вмешательства (t ≥ 1910)")
    ax_p.set_xlabel("Заяц (тыс.)")
    ax_p.set_ylabel("Рысь (тыс.)")
    ax_p.grid(alpha=0.3)
    ax_p.legend(fontsize=8)
    # равновесия
    for (f, lam, xs, ys, *_), c in zip(rows, COLORS):
        ax_p.plot(xs, ys, "x", color=c, ms=9, mew=2)
    plt.tight_layout()
    p1 = OUT / "intervention_love_trajectories.png"
    plt.savefig(p1, dpi=140)
    plt.close(fig)

    # ---------------- рисунок 2: DAG с новыми вершинами ----------------
    tree = model.built["tree"]
    pos = layout_layered(tree)
    # короткие подписи: у глубоких узлов развёрнутое выражение не помещается в круг
    import visualize_tree as vt
    vt.node_label = lambda n: (vt.OP_SYMBOL.get(n.op, n.op) if n.op else
                               (f"{n.name}\n={n.value:g}" if n.value is not None else n.name))
    fig2, ax = plt.subplots(figsize=(13, 8))
    draw_tree(ax, tree, pos, highlight_shared=model.built["new_ids"],
              title="DAG с интервенцией: розовым обведены новые вершины "
                    "(u, lam, lam·u, love=lam·u·x·y и два новых корня «+»)")
    plt.tight_layout()
    p2 = OUT / "lotka_volterra_tree_with_love.png"
    plt.savefig(p2, dpi=130, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nГрафики: {p1}\n         {p2}")


if __name__ == "__main__":
    main()
