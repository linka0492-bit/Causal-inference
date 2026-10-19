"""
Интервенция «добавление новых объектов» (способ 6 из списка руководителя).

К 21 наблюдению Hudson Bay добавляем k новых наблюдений в промежуточные моменты
времени (t = 0.5, 1.5, ..., 19.5). Значения новых точек генерируем из модели
Volterra-Lotka (параметры LM по полной выборке) + шум с std остатков подгонки:
  add_same     -- 5 точек «из того же распределения» (КОНТРОЛЬ: параметры не должны
                  значимо поменяться);
  add_shift5   -- 5 точек, у которых заяц умножен на 1.5 (слегка другое распределение);
  add_shift10  -- то же, 10 точек.
Дальше та же мини-батч процедура (100 батчей по 12 точек из 21+k), те же расстояния и
КС-тест, что и в distribution_distances.py; сравнение с base.
"""

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lv_common import LVModel, fit, T_OBS, DATA, PARAM_NAMES, LOWER, UPPER
from minibatch_distribution import make_minibatches, LABELS
from distribution_distances import metrics, holm, load

OUT = Path(__file__).resolve().parent / "outputs"
N_BATCHES, BATCH = 100, 12
HALF_TIMES = np.arange(0.5, 20.0, 1.0)


def make_data(model, p_full, k, shift, rng):
    """Возвращает (времена, данные) с k добавленными точками."""
    resid = model.simulate(p_full, T_OBS) - DATA
    sd = resid.std(axis=0, ddof=1)
    tn = np.sort(rng.choice(HALF_TIMES, size=k, replace=False))
    new = model.simulate(p_full, np.concatenate([[0.0], tn]))[1:]  # t_eval[0] должен быть 0 (старт)
    new = new + rng.normal(0, sd, size=new.shape)
    new[:, 0] *= shift
    new = np.clip(new, 0.0, None)
    t_all = np.concatenate([T_OBS, tn])
    d_all = np.vstack([DATA, new])
    order = np.argsort(t_all)
    return t_all[order], d_all[order], tn, new


def fit_batches(model, t, data, p0, rng):
    batches = make_minibatches(len(t), BATCH, N_BATCHES, rng)
    est = []
    for idx in batches:
        def resid(p):
            return (model.simulate(p, t) - data)[idx].ravel()
        res = least_squares(resid, np.clip(p0, LOWER * 1.0001, UPPER * 0.9999),
                            method="trf", bounds=(LOWER, UPPER), max_nfev=25)
        est.append(res.x)
    return np.array(est)


def main():
    rng = np.random.default_rng(5)
    model = LVModel(rtol=1e-6, atol=1e-6)
    p_full, _, _ = fit(model)
    base = load("base")

    scen = {"add_same": (5, 1.0), "add_shift5": (5, 1.5), "add_shift10": (10, 1.5)}
    samples, results, added = {}, {}, {}
    for name, (k, shift) in scen.items():
        t, d, tn, new = make_data(model, p_full, k, shift, rng)
        added[name] = (tn, new)
        est = fit_batches(model, t, d, p_full, rng)
        samples[name] = est
        np.savetxt(OUT / f"minibatch_estimates_{name}.csv", est, delimiter=",",
                   header=",".join(PARAM_NAMES), comments="")
        rows = [metrics(base[:, j], est[:, j]) for j in range(6)]
        adj = holm(np.array([r["ks_p"] for r in rows]))
        for r, pa in zip(rows, adj):
            r["ks_p_holm"] = float(pa)
        results[name] = rows
        print(f"{name} готово: k={k}, shift={shift}")

    print("\nСреднее ± std по батчам")
    print(f"  {'сценарий':12s}" + "".join(f"{p:>16s}" for p in PARAM_NAMES))
    for nm, arr in [("base", base)] + list(samples.items()):
        print(f"  {nm:12s}" + "".join(f"{arr[:, j].mean():8.3g}±{arr[:, j].std(ddof=1):<7.2g}" for j in range(6)))

    print("\nКС: D / p (Холм), W1/std(base), JS")
    for nm in scen:
        print(f"  {nm}")
        for pn, r in zip(PARAM_NAMES, results[nm]):
            p = r["ks_p_holm"]
            print(f"    {pn:6s} D={r['ks_D']:.2f} p={'<1e-3' if p < 1e-3 else f'{p:.3f}':>6s} "
                  f"W1/σ={r['w1_norm']:.2f} JS={r['js']:.2f}")

    with open(OUT / "add_objects_distances.csv", "w", encoding="utf-8") as f:
        f.write("scenario,param,ks_D,ks_p,ks_p_holm,w1_norm,js,kl\n")
        for nm in scen:
            for pn, r in zip(PARAM_NAMES, results[nm]):
                f.write(f"{nm},{pn},{r['ks_D']:.4f},{r['ks_p']:.3g},{r['ks_p_holm']:.3g},"
                        f"{r['w1_norm']:.4f},{r['js']:.4f},{r['kl']:.4f}\n")

    # ---- рисунок 1: данные + добавленные точки ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    tt = np.linspace(0, 20, 400)
    sim = model.simulate(p_full, tt)
    for ax, i, nm in ((axes[0], 0, "Заяц"), (axes[1], 1, "Рысь")):
        ax.plot(1900 + tt, sim[:, i], color="gray", lw=1, label="модель (LM, полная выборка)")
        ax.scatter(1900 + T_OBS, DATA[:, i], color="black", s=22, label="данные Hudson Bay", zorder=5)
        for c, nm2 in zip(["tab:green", "tab:red"], ["add_same", "add_shift10"]):
            tn, new = added[nm2]
            ax.scatter(1900 + tn, new[:, i], color=c, marker="^", s=40, label=f"добавлено: {nm2}", zorder=6)
        ax.set_title(nm)
        ax.set_xlabel("год")
        ax.set_ylabel("численность (тыс.)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "add_objects_data.png", dpi=140)
    plt.close(fig)

    # ---- рисунок 2: распределения параметров ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    cols = {"base": "black", "add_same": "tab:green", "add_shift5": "tab:orange", "add_shift10": "tab:red"}
    allsets = {"base": base, **samples}
    for j, ax in enumerate(axes.ravel()):
        allv = np.concatenate([a[:, j] for a in allsets.values()])
        lo, hi = np.percentile(allv, [1, 99])
        bins = np.linspace(lo, hi, 30)
        for nm, arr in allsets.items():
            star = ""
            if nm != "base":
                star = " *" if results[nm][j]["ks_p_holm"] < 0.05 else " (н/з)"
            ax.hist(arr[:, j], bins=bins, density=True, histtype="step", lw=2.2 if nm == "base" else 1.5,
                    color=cols[nm], label=nm + star)
        ax.set_title(LABELS[j])
        ax.set_xlabel("значение параметра")
        ax.set_ylabel("плотность")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Интервенция «добавление новых объектов»: распределение параметров "
                 "(* — значимо по КС, н/з — не значимо)")
    plt.tight_layout()
    plt.savefig(OUT / "add_objects_hist.png", dpi=140)
    plt.close(fig)
    print(f"\nГотово: {OUT}/add_objects_*.png/.csv")


if __name__ == "__main__":
    main()
