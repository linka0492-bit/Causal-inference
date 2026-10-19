"""
Интервенции «зашумить со смещением» и «занулить со смещением» (способ 4 из списка).

Шум со смещением: к обеим сериям добавляется N(mu*std_k, (0.3*std_k)^2), где std_k --
std k-й серии, mu -- величина смещения в долях std. mu=0 -- КОНТРОЛЬ (шум без смещения),
по нему видно, что даёт именно смещение, а не сам шум.
  bias_m025 (mu=-0.25), bias_0 (mu=0), bias_p025, bias_p05, bias_p10.
Зануление со смещением: серия рыси заменяется константой, НЕ равной её среднему:
  const_x05 (0.5*среднее), const_x20 (2*среднее).  (const на уровне среднего = zero_lynx из прошлого шага.)

Данные обрезаются снизу на 0.1 (численность не может быть отрицательной); доля обрезанных
значений печатается. Дальше: мини-батч процедура (100 батчей по 12 из 21), те же метрики и КС.
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
FLOOR = 0.1
SD = DATA.std(axis=0, ddof=1)
MEAN = DATA.mean(axis=0)

BIAS = {"bias_m025": -0.25, "bias_0": 0.0, "bias_p025": 0.25, "bias_p05": 0.5, "bias_p10": 1.0}
CONST = {"const_x05": 0.5, "const_x20": 2.0}


def make(name, rng):
    d = DATA.copy()
    if name in BIAS:
        d += rng.normal(BIAS[name] * SD, 0.3 * SD, size=d.shape)
    else:
        d[:, 1] = CONST[name] * MEAN[1]
    clipped = float(np.mean(d < FLOOR))
    return np.clip(d, FLOOR, None), clipped


def fit_batches(model, data, p0, rng):
    batches = make_minibatches(len(T_OBS), BATCH, N_BATCHES, rng)
    est = []
    for idx in batches:
        def resid(p):
            return (model.simulate(p, T_OBS) - data)[idx].ravel()
        res = least_squares(resid, np.clip(p0, LOWER * 1.0001, UPPER * 0.9999),
                            method="trf", bounds=(LOWER, UPPER), max_nfev=25)
        est.append(res.x)
    return np.array(est)


def main():
    rng = np.random.default_rng(11)
    model = LVModel(rtol=1e-6, atol=1e-6)
    p_full, _, _ = fit(model)
    base = load("base")

    names = list(BIAS) + list(CONST)
    samples, results, datasets = {}, {}, {}
    for nm in names:
        data, clipped = make(nm, rng)
        datasets[nm] = data
        est = fit_batches(model, data, p_full, rng)
        samples[nm] = est
        np.savetxt(OUT / f"minibatch_estimates_{nm}.csv", est, delimiter=",",
                   header=",".join(PARAM_NAMES), comments="")
        rows = [metrics(base[:, j], est[:, j]) for j in range(6)]
        adj = holm(np.array([r["ks_p"] for r in rows]))
        for r, pa in zip(rows, adj):
            r["ks_p_holm"] = float(pa)
        results[nm] = rows
        # доля параметров на границе допустимой области -- признак «сломанной» подгонки
        at_bound = float(np.mean((est <= LOWER * 1.01) | (est >= UPPER * 0.99)))
        print(f"{nm:10s} готово, обрезано значений данных: {100 * clipped:.0f}%, "
              f"оценок на границе: {100 * at_bound:.0f}%")

    print("\nСреднее ± std по батчам")
    print(f"  {'сценарий':10s}" + "".join(f"{p:>16s}" for p in PARAM_NAMES))
    for nm, arr in [("base", base)] + list(samples.items()):
        print(f"  {nm:10s}" + "".join(f"{arr[:, j].mean():8.3g}±{arr[:, j].std(ddof=1):<7.2g}" for j in range(6)))

    print("\nКС D (p Холма: * значимо), W1/σ(base), JS")
    print(f"  {'сценарий':10s}" + "".join(f"{p:>22s}" for p in PARAM_NAMES))
    for nm in names:
        cells = []
        for r in results[nm]:
            star = "*" if r["ks_p_holm"] < 0.05 else " "
            cells.append(f"D={r['ks_D']:.2f}{star} W={r['w1_norm']:5.1f} J={r['js']:.2f}")
        print(f"  {nm:10s}" + "".join(f"{c:>22s}" for c in cells))

    with open(OUT / "bias_distances.csv", "w", encoding="utf-8") as f:
        f.write("scenario,param,ks_D,ks_p,ks_p_holm,w1_norm,js,kl\n")
        for nm in names:
            for pn, r in zip(PARAM_NAMES, results[nm]):
                f.write(f"{nm},{pn},{r['ks_D']:.4f},{r['ks_p']:.3g},{r['ks_p_holm']:.3g},"
                        f"{r['w1_norm']:.4f},{r['js']:.4f},{r['kl']:.4f}\n")

    # ---- рисунок 1: данные после интервенций ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    show = ["bias_m025", "bias_0", "bias_p10", "const_x05", "const_x20"]
    cmap = dict(zip(show, ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:brown"]))
    for ax, i, title in ((axes[0], 0, "Заяц"), (axes[1], 1, "Рысь")):
        ax.plot(1900 + T_OBS, DATA[:, i], "k-o", ms=4, lw=2, label="исходные данные")
        for nm in show:
            ax.plot(1900 + T_OBS, datasets[nm][:, i], "-", lw=1, alpha=0.8, color=cmap[nm], label=nm)
        ax.set_title(title)
        ax.set_xlabel("год")
        ax.set_ylabel("численность (тыс.)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "bias_data.png", dpi=140)
    plt.close(fig)

    # ---- рисунок 2: сдвиг среднего параметра и W1 от величины смещения ----
    mus = [BIAS[n] for n in BIAS]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    bnames = list(BIAS)
    for j, ax in enumerate(axes.ravel()):
        means = np.array([samples[n][:, j].mean() for n in bnames])
        sds = np.array([samples[n][:, j].std(ddof=1) for n in bnames])
        ax.errorbar(mus, means, yerr=sds, fmt="o-", color="tab:blue", capsize=3, label="среднее ± std по батчам")
        ax.axhline(base[:, j].mean(), color="black", ls="--", label="base (без интервенции)")
        for mu, n in zip(mus, bnames):
            if results[n][j]["ks_p_holm"] < 0.05:
                ax.plot(mu, samples[n][:, j].mean(), "r*", ms=14)
        ax.set_title(LABELS[j])
        ax.set_xlabel("смещение шума, в долях std серии")
        ax.set_ylabel("значение параметра")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Шум со смещением: оценки параметров от величины смещения (красная звезда — значимо отличается от base по КС)")
    plt.tight_layout()
    plt.savefig(OUT / "bias_sweep.png", dpi=140)
    plt.close(fig)

    # ---- рисунок 3: гистограммы константы со смещением ----
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    allsets = {"base": base, "const_x05": samples["const_x05"], "const_x20": samples["const_x20"]}
    cols = {"base": "black", "const_x05": "tab:purple", "const_x20": "tab:brown"}
    for j, ax in enumerate(axes.ravel()):
        allv = np.concatenate([a[:, j] for a in allsets.values()])
        bins = np.linspace(*np.percentile(allv, [0, 100]), 40)
        for nm, arr in allsets.items():
            ax.hist(arr[:, j], bins=bins, density=True, histtype="step", lw=2.2 if nm == "base" else 1.5,
                    color=cols[nm], label=nm)
        ax.set_title(LABELS[j])
        ax.set_xlabel("значение параметра")
        ax.set_ylabel("плотность")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Зануление со смещением: серия рыси заменена константой на уровне 0.5× и 2× среднего")
    plt.tight_layout()
    plt.savefig(OUT / "bias_const_hist.png", dpi=140)
    plt.close(fig)
    print(f"\nГотово: {OUT}/bias_*.png/.csv")


if __name__ == "__main__":
    main()
