"""
1) Переносим готовые выборки Лотки-Вольтерры в общее хранилище (hist_store).
2) Прогоняем ту же схему на второй модели (Ван-дер-Поль, синтетика с известной истиной):
   base + интервенции в данные -> мини-батч оценки -> хранилище (с гиперпараметрами).
3) Сравниваем модели: расстояния до/после интервенции, КС-тест, сводные картинки.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lv_common  # noqa: F401
from lv_common import PARAM_NAMES as LV_NAMES
from minibatch_distribution import make_minibatches
from hist_store import HistStore
from models_vdp import (VdPModel, synthetic_data, VDP_NAMES, TRUTH, LOWER, UPPER, P_START, T_GRID)
from distribution_fits import FAMILIES

OUT = Path(__file__).resolve().parent / "outputs"
STORE = HistStore(OUT / "hist_store")

N_BATCHES, BATCH, BATCH_SMALL = 100, 20, 8   # VdP: 20 из 60 (~как 12 из 21), малый батч 8

# имена сценариев: общие (для сравнения моделей) <- имя в LV
ALIAS_LV = {"zero_second": "zero_lynx", "noise": "noise", "noise_cond": "noise_cond",
            "shuffle30": "shuffle30", "small_sample": "small_sample", "bias_p05": "bias_p05"}


# ---------------------------------------------------------------- 1) миграция LV
def migrate_lv():
    n = 0
    for p in sorted(OUT.glob("minibatch_estimates_*.csv")):
        sc = p.stem.replace("minibatch_estimates_", "")
        arr = np.loadtxt(p, delimiter=",", skiprows=1)
        STORE.put("lotka_volterra", sc, LV_NAMES, arr,
                  meta={"source": "migrated", "data": "Hudson Bay 1900-1920", "file": p.name})
        n += 1
    print(f"Перенесено в хранилище (lotka_volterra): {n} сценариев")


# ---------------------------------------------------------------- 2) VdP
def make_scenario(name, base, rng):
    d = base.copy()
    sd = base.std(axis=0, ddof=1)
    if name == "zero_second":
        d[:, 1] = d[:, 1].mean()
    elif name == "noise":
        d += rng.normal(0, 0.3 * sd, size=d.shape)
    elif name == "noise_cond":
        d[:, 0] += rng.normal(0, 1, len(d)) * 0.3 * np.abs(d[:, 1])
    elif name == "shuffle30":
        idx = rng.choice(len(d), size=int(0.3 * len(d)), replace=False)
        d[idx] = d[idx][:, ::-1]
    elif name == "bias_p05":
        d += rng.normal(0.5 * sd, 0.3 * sd, size=d.shape)
    return d


def fit_batches(model, t, data, p0, batch, rng):
    est = []
    for idx in make_minibatches(len(t), batch, N_BATCHES, rng):
        def resid(p):
            return (model.simulate(p, t) - data)[idx].ravel()
        res = least_squares(resid, np.clip(p0, LOWER + 1e-6, UPPER - 1e-6),
                            method="trf", bounds=(LOWER, UPPER), max_nfev=25)
        est.append(res.x)
    return np.array(est)


def run_vdp():
    rng = np.random.default_rng(7)
    model = VdPModel()
    data = synthetic_data(model)
    res = least_squares(lambda p: (model.simulate(p, T_GRID) - data).ravel(), P_START,
                        method="trf", bounds=(LOWER, UPPER))
    p_full = res.x
    print("VdP: истина  ", TRUTH)
    print("VdP: подгонка", np.round(p_full, 3), f"RSS={np.sum(res.fun**2):.3f} "
          f"(ожидаемо ~ n*sigma^2 = {data.size * 0.15**2:.2f})")
    np.savetxt(OUT / "vdp_synthetic_data.csv", np.column_stack([T_GRID, data]), delimiter=",",
               header="t,x,v", comments="")

    scenarios = ["base", "zero_second", "noise", "noise_cond", "shuffle30", "small_sample", "bias_p05"]
    for sc in scenarios:
        d = data if sc == "base" else make_scenario(sc, data, rng)
        b = BATCH_SMALL if sc == "small_sample" else BATCH
        est = fit_batches(model, T_GRID, d, p_full, b, rng)
        at_bound = float(np.mean((est <= LOWER * 1.01 + 1e-9) | (est >= UPPER * 0.99)))
        STORE.put("van_der_pol", sc, VDP_NAMES, est,
                  meta={"data": "synthetic", "truth": TRUTH.tolist(), "noise_sd": 0.15,
                        "batch_size": b, "n_points": len(T_GRID), "frac_at_bound": at_bound})
        print(f"  van_der_pol/{sc:13s} готово (batch={b}, на границе {100 * at_bound:.0f}%)")
    return p_full


# ---------------------------------------------------------------- 3) сравнение
def report():
    print("\nХранилище:")
    for m in STORE.models():
        print(f"  {m}: {', '.join(STORE.scenarios(m))}")

    # ---- VdP base: покрытие истины ----
    names, base = STORE.get_samples("van_der_pol", "base")
    print("\nVdP base: среднее ± std по батчам, истина, попадание истины в 95% интервал выборки")
    for j, nm in enumerate(names):
        lo, hi = np.percentile(base[:, j], [2.5, 97.5])
        inside = lo <= TRUTH[j] <= hi
        print(f"  {nm:6s} {base[:, j].mean():8.4f} ± {base[:, j].std(ddof=1):.4f}  истина={TRUTH[j]:6.3f}  "
              f"[{lo:.3f}, {hi:.3f}]  {'внутри' if inside else 'СНАРУЖИ'}")

    # ---- сравнение сценариев с base в обеих моделях ----
    summary = {}  # (model, generic_name) -> rows
    for gen, lv_name in ALIAS_LV.items():
        summary[("lotka_volterra", gen)] = STORE.compare("lotka_volterra", "base", lv_name)
        summary[("van_der_pol", gen)] = STORE.compare("van_der_pol", "base", gen)

    print("\nСравнение с base (среднее по параметрам): W1/σ | JS | число значимых по КС (Холм)")
    print(f"  {'сценарий':14s}" + "".join(f"{m:>34s}" for m in ("lotka_volterra", "van_der_pol")))
    for gen in ALIAS_LV:
        cells = []
        for m in ("lotka_volterra", "van_der_pol"):
            rows = summary[(m, gen)]
            w = np.mean([r["w1_norm"] for r in rows])
            js = np.mean([r["js"] for r in rows])
            sig = sum(r["ks_p_holm"] < 0.05 for r in rows)
            cells.append(f"W1/σ={w:6.2f} JS={js:.2f} знач={sig}/{len(rows)}")
        print(f"  {gen:14s}" + "".join(f"{c:>34s}" for c in cells))

    with open(OUT / "cross_model_distances.csv", "w", encoding="utf-8") as f:
        f.write("model,scenario,param,ks_D,ks_p_holm,w1_norm,js,kl\n")
        for (m, gen), rows in summary.items():
            for r in rows:
                f.write(f"{m},{gen},{r['param']},{r['ks_D']:.4f},{r['ks_p_holm']:.3g},"
                        f"{r['w1_norm']:.4f},{r['js']:.4f},{r['kl']:.4f}\n")

    # ---- рисунок 1: тепловые карты W1/σ по моделям + столбики ----
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    for k, (m, title) in enumerate((("lotka_volterra", "Лотка–Вольтерра (Hudson Bay)"),
                                    ("van_der_pol", "Ван-дер-Поль (синтетика)"))):
        ax = fig.add_subplot(gs[0, k])
        pn = [r["param"] for r in summary[(m, "noise")]]
        M = np.array([[r["w1_norm"] for r in summary[(m, g)]] for g in ALIAS_LV])
        im = ax.imshow(np.log10(M + 0.1), cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pn)))
        ax.set_xticklabels(pn)
        ax.set_yticks(range(len(ALIAS_LV)))
        ax.set_yticklabels(list(ALIAS_LV))
        for i, g in enumerate(ALIAS_LV):
            for j in range(len(pn)):
                sig = summary[(m, g)][j]["ks_p_holm"] < 0.05
                ax.text(j, i, f"{M[i, j]:.1f}" + ("*" if sig else ""), ha="center", va="center",
                        fontsize=8, color="white" if np.log10(M[i, j] + 0.1) < 0.9 else "black")
        ax.set_title(f"{title}\nW1/σ(base), цвет — log10 (* значимо по КС)", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = fig.add_subplot(gs[1, :])
    xs = np.arange(len(ALIAS_LV))
    for k, (m, c) in enumerate((("lotka_volterra", "tab:blue"), ("van_der_pol", "tab:orange"))):
        vals = [np.mean([r["w1_norm"] for r in summary[(m, g)]]) for g in ALIAS_LV]
        ax.bar(xs + (k - 0.5) * 0.36, vals, 0.36, color=c, label=m)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(list(ALIAS_LV))
    ax.set_ylabel("среднее по параметрам W1/σ(base)")
    ax.set_title("Насколько интервенция сдвигает распределение параметров: сравнение моделей")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "cross_model_comparison.png", dpi=140)
    plt.close(fig)

    # ---- рисунок 2: VdP base, гистограммы + подгонки + истина ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    colors = {"norm": "tab:red", "t": "tab:purple", "laplace": "tab:green",
              "lognorm": "tab:orange", "gamma": "tab:brown"}
    info = STORE.get_hyper("van_der_pol", "base")
    for j, ax in enumerate(axes):
        s = base[:, j]
        ax.hist(s, bins=22, density=True, color="lightsteelblue", edgecolor="white")
        xs_ = np.linspace(s.min(), s.max(), 300)
        rec = info["params"][names[j]]
        for k, v in rec["fits"].items():
            ax.plot(xs_, FAMILIES[k].pdf(xs_, *v["params"]), color=colors[k],
                    lw=2 if k == rec["best_by_aic"] else 1, label=k)
        ax.axvline(TRUTH[j], color="black", ls="--", lw=2, label="истина")
        ax.set_title(names[j])
        ax.set_xlabel("значение параметра")
        ax.set_ylabel("плотность")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Ван-дер-Поль: распределение параметров по мини-батчам и аппроксимации (пунктир — истинное значение)")
    plt.tight_layout()
    plt.savefig(OUT / "vdp_base_hist.png", dpi=140)
    plt.close(fig)

    # ---- рисунок 3: синтетические данные и подгонка ----
    model = VdPModel()
    d = np.loadtxt(OUT / "vdp_synthetic_data.csv", delimiter=",", skiprows=1)
    pm = np.array([STORE.get_hyper("van_der_pol", "base")["params"][n]["mean"] for n in VDP_NAMES])
    tt = np.linspace(0, 20, 400)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(d[:, 0], d[:, 1], s=14, color="tab:blue", label="x (синтетика, шум σ=0.15)")
    ax.scatter(d[:, 0], d[:, 2], s=14, color="tab:orange", label="v (синтетика)")
    ax.plot(tt, model.simulate(TRUTH, tt)[:, 0], "k--", lw=1, label="истинная модель")
    ax.plot(tt, model.simulate(pm, tt)[:, 0], color="tab:blue", lw=1.2, label="среднее по батчам, x")
    ax.plot(tt, model.simulate(pm, tt)[:, 1], color="tab:orange", lw=1.2, label="среднее по батчам, v")
    ax.set_xlabel("t")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Ван-дер-Поль: синтетические данные и модель с усреднёнными параметрами")
    plt.tight_layout()
    plt.savefig(OUT / "vdp_data.png", dpi=140)
    plt.close(fig)
    print(f"\nГотово: {OUT}/hist_store/, cross_model_comparison.png, vdp_base_hist.png, vdp_data.png, "
          f"cross_model_distances.csv")


if __name__ == "__main__":
    if "--report-only" not in sys.argv:
        migrate_lv()
        run_vdp()
    report()
