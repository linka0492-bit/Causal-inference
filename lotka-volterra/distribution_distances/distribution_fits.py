"""
Байесовские гиперпараметры: аппроксимация выборки параметров (из мини-батч процедуры)
одномерными распределениями + интервенции в данные.

Вход: выборка вектора параметров (alpha, beta, gamma, delta, x0, y0), полученная
мини-батч процедурой (на каждом батче -- своя подгонка). Каждый параметр разбираем
как ОДНОМЕРНУЮ случайную величину (договорённость с руководителем).
Выход: для каждого сценария и параметра -- гиперпараметры нескольких семейств
распределений + AIC + статистика Колмогорова-Смирнова; всё в hyperparams.json.

Семейства (предложение на согласование):
  norm     -- нормальное (базовое);
  t        -- Стьюдента (тяжёлые хвосты: выбросы батчей);
  laplace  -- Лапласа (острая вершина, экспоненциальные хвосты);
  lognorm  -- логнормальное (положительные параметры, правая скошенность);
  gamma    -- гамма (положительные параметры, скошенность).

Сценарии интервенций в данные (21 точка заяц/рысь):
  base          -- без вмешательства;
  zero_lynx     -- рысь заменена константой (среднее);
  noise         -- шум N(0,(0.3*std)^2) в обе серии, без учёта другой переменной;
  noise_cond    -- шум зайца с sd, пропорциональной значению рыси (с учётом);
  shuffle30     -- у 30% моментов времени заяц и рысь поменяны местами;
  small_sample  -- «сокращение выборки»: батчи по 5 точек вместо 12.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lv_common import LVModel, T_OBS, DATA, PARAM_NAMES, LOWER, UPPER
from minibatch_distribution import make_minibatches, LABELS

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

FAMILIES = {
    "norm": stats.norm,
    "t": stats.t,
    "laplace": stats.laplace,
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
}
N_BATCHES = 100
BATCH = 12


# ----------------------------- интервенции в данные -----------------------------

def intervene(name, data, rng):
    d = data.copy()
    if name == "zero_lynx":
        d[:, 1] = d[:, 1].mean()
    elif name == "noise":
        d += rng.normal(0, 0.3 * d.std(axis=0), size=d.shape)
    elif name == "noise_cond":
        d[:, 0] += rng.normal(0, 1, len(d)) * 0.5 * d[:, 1]
    elif name == "shuffle30":
        k = int(0.3 * len(d))
        idx = rng.choice(len(d), size=k, replace=False)
        d[idx] = d[idx][:, ::-1]
    return np.clip(d, 0.0, None)


# ----------------------------- подгонка по батчам -----------------------------

def fit_batches(model, data, p0, batch, rng):
    batches = make_minibatches(len(T_OBS), batch, N_BATCHES, rng)
    est = []
    for idx in batches:
        def resid(p):
            return (model.simulate(p, T_OBS) - data)[idx].ravel()
        res = least_squares(resid, np.clip(p0, LOWER * 1.0001, UPPER * 0.9999),
                            method="trf", bounds=(LOWER, UPPER), max_nfev=25)
        est.append(res.x)
    return np.array(est)


# ----------------------------- аппроксимация распределениями -----------------------------

def fit_families(s):
    """Возвращает {семейство: {params, loglik, aic, ks}}; для гаммы и логнормального loc=0."""
    out = {}
    for name, dist in FAMILIES.items():
        try:
            if name in ("lognorm", "gamma"):
                if np.any(s <= 0):
                    continue
                params = dist.fit(s, floc=0)
            else:
                params = dist.fit(s)
            ll = float(np.sum(dist.logpdf(s, *params)))
            k = len(params) - (1 if name in ("lognorm", "gamma") else 0)
            ks = float(stats.kstest(s, dist.cdf, args=params).statistic)
            if not np.isfinite(ll):
                continue
            out[name] = {"params": [float(v) for v in params], "loglik": ll,
                         "aic": 2 * k - 2 * ll, "ks_stat": ks}
        except Exception:  # семейство не подошло -- пропускаем
            continue
    return out


def main():
    rng = np.random.default_rng(0)
    model = LVModel(rtol=1e-6, atol=1e-6)

    # старт -- оптимум LM по полной (неискажённой) выборке
    from lv_common import fit
    p_full, rss, _ = fit(model)
    print("Старт (LM по полной выборке):", np.round(p_full, 4), f"RSS={rss:.1f}\n")

    scenarios = ["base", "zero_lynx", "noise", "noise_cond", "shuffle30", "small_sample"]
    samples, store = {}, {"model": "lotka_volterra", "n_batches": N_BATCHES, "scenarios": {}}

    for sc in scenarios:
        data = DATA if sc == "base" else intervene(sc, DATA, rng)
        batch = 5 if sc == "small_sample" else BATCH
        est = fit_batches(model, data, p_full, batch, rng)
        samples[sc] = est
        np.savetxt(OUT / f"minibatch_estimates_{sc}.csv", est, delimiter=",",
                   header=",".join(PARAM_NAMES), comments="")
        store["scenarios"][sc] = {"batch_size": batch, "params": {}}
        for j, pn in enumerate(PARAM_NAMES):
            fits = fit_families(est[:, j])
            best = min(fits, key=lambda k: fits[k]["aic"]) if fits else None
            store["scenarios"][sc]["params"][pn] = {
                "mean": float(est[:, j].mean()), "std": float(est[:, j].std(ddof=1)),
                "best_by_aic": best, "fits": fits}
        print(f"{sc:13s} готово (batch={batch})")

    with open(OUT / "hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)

    # ---- таблица AIC для базового сценария ----
    print("\nBASE: AIC по семействам (меньше -- лучше), * = лучшее")
    print("  param   " + "".join(f"{k:>10s}" for k in FAMILIES))
    for pn in PARAM_NAMES:
        info = store["scenarios"]["base"]["params"][pn]
        row = ""
        for k in FAMILIES:
            v = info["fits"].get(k)
            cell = "-" if v is None else f"{v['aic']:.1f}" + ("*" if k == info["best_by_aic"] else "")
            row += f"{cell:>10s}"
        print(f"  {pn:7s}" + row)

    print("\nBASE: гиперпараметры (norm: mu, sigma; t: df, loc, scale)")
    for pn in PARAM_NAMES:
        f = store["scenarios"]["base"]["params"][pn]["fits"]
        print(f"  {pn:6s} norm mu={f['norm']['params'][0]:.4g} sigma={f['norm']['params'][1]:.3g} | "
              f"t df={f['t']['params'][0]:.1f}")

    # ---- рисунок 1: base + подогнанные плотности ----
    est = samples["base"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    colors = {"norm": "tab:red", "t": "tab:purple", "laplace": "tab:green",
              "lognorm": "tab:orange", "gamma": "tab:brown"}
    for j, ax in enumerate(axes.ravel()):
        s = est[:, j]
        ax.hist(s, bins=25, density=True, color="lightsteelblue", edgecolor="white")
        xs = np.linspace(s.min(), s.max(), 400)
        info = store["scenarios"]["base"]["params"][PARAM_NAMES[j]]
        for k, v in info["fits"].items():
            ax.plot(xs, FAMILIES[k].pdf(xs, *v["params"]), color=colors[k], lw=2 if k == info["best_by_aic"] else 1.1,
                    label=f"{k}" + (" (лучшее по AIC)" if k == info["best_by_aic"] else ""))
        ax.set_title(LABELS[j])
        ax.set_xlabel("значение параметра")
        ax.set_ylabel("плотность")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"Аппроксимация распределения параметров ({N_BATCHES} мини-батчей по {BATCH} точек)")
    plt.tight_layout()
    plt.savefig(OUT / "param_fits_base.png", dpi=140)
    plt.close(fig)

    # ---- рисунок 2: сценарии интервенций друг на друге ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    sc_colors = dict(zip(scenarios, ["black", "tab:red", "tab:blue", "tab:cyan", "tab:green", "tab:orange"]))
    for j, ax in enumerate(axes.ravel()):
        allv = np.concatenate([samples[sc][:, j] for sc in scenarios])
        lo, hi = np.percentile(allv, [1, 99])
        bins = np.linspace(lo, hi, 35)
        for sc in scenarios:
            ax.hist(samples[sc][:, j], bins=bins, density=True, histtype="step", lw=2 if sc == "base" else 1.4,
                    color=sc_colors[sc], label=sc)
        ax.set_title(LABELS[j])
        ax.set_xlabel("значение параметра")
        ax.set_ylabel("плотность")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Распределение параметров после интервенций в данные (оси обрезаны по 1–99 перцентилям)")
    plt.tight_layout()
    plt.savefig(OUT / "param_hist_interventions.png", dpi=140)
    plt.close(fig)

    # ---- краткая сводка по сценариям ----
    print("\nСценарии: среднее ± std по батчам")
    print("  scenario      " + "".join(f"{p:>16s}" for p in PARAM_NAMES))
    for sc in scenarios:
        row = "".join(f"{samples[sc][:, j].mean():8.3g}±{samples[sc][:, j].std(ddof=1):<7.2g}" for j in range(6))
        print(f"  {sc:13s} {row}")
    print(f"\nГотово: {OUT}/hyperparams.json, param_fits_base.png, param_hist_interventions.png")


if __name__ == "__main__":
    main()
