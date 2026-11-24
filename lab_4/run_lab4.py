"""Полный эксперимент ЛР № 4: python run_lab4.py --out results --n 40 --seed 0"""
import argparse, json, os, time
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

from scmforecast.experiment import (Dataset, tune, fit_baselines, leave_out, error_by_intervention, param_error,
                                    DEFAULT_GRID)
from scmforecast.forecaster import LinearGraphForecaster, Prepared
from scmforecast.objective import evaluate
from scmforecast.systems import hudson_bay
from scmforecast import viz


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=40, help="экземпляров каждой системы")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="урезанная сетка гиперпараметров")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    J = lambda r: {k: v for k, v in r.items() if k != "per_sample"}

    log("коллекция"); ds = Dataset(a.n, a.seed)
    tr, va, te = ds.split(seed=a.seed)
    log(f"выборок: {len(ds.samples)}; train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    viz.plot_collection(ds.samples, fname=f"{a.out}/fig_collection.png")

    log("настройка гиперпараметров по J")
    grid = dict(ridge=(1.0,), tau=(0.5,), max_flips=(3,), final_prune=(True,)) if a.quick else DEFAULT_GRID
    rows, model = tune(ds.train_preps(tr), ds.evalset(va, seed=a.seed + 1), grid, seed=a.seed)
    json.dump(rows, open(f"{a.out}/tuning.json", "w"), indent=1)
    viz.plot_tuning(rows, fname=f"{a.out}/fig_tuning.png")
    best = rows[0]
    params = dict(ridge=best["ridge"], tau=best["tau"], max_flips=best["max_flips"], final_prune=best["final_prune"])
    log("лучшие гиперпараметры:", params, "J(val) =", round(best["J"], 3))
    model.save(f"{a.out}/forecaster.json")

    log("тест: сравнение методов")
    es_te = ds.evalset(te, seed=a.seed + 2)
    base = fit_baselines(ds.train_preps(tr))
    test = {"линейная модель": evaluate(model.predict, es_te)}
    for k, b in base.items():
        if b is not None:
            test[k] = evaluate(b.predict, es_te)
    for k, r in test.items():
        log(k, {x: round(y, 3) for x, y in J(r).items()})
    json.dump({k: J(v) for k, v in test.items()}, open(f"{a.out}/test_metrics.json", "w"), indent=1, ensure_ascii=False)
    viz.plot_methods({k: v for k, v in test.items()}, "exact", "тест: доля точно угаданных структур", f"{a.out}/fig_methods_exact.png")
    viz.plot_per_system(test["линейная модель"]["per_sample"], f"{a.out}/fig_per_system.png")

    log("устойчивость к интервенциям в данные")
    inter = {"линейная модель": error_by_intervention(model.predict, [ds.samples[i] for i in te])}
    for k, b in base.items():
        if b is not None:
            inter[k] = error_by_intervention(b.predict, [ds.samples[i] for i in te])
    json.dump(inter, open(f"{a.out}/interventions.json", "w"), indent=1, ensure_ascii=False)
    names = list(next(iter(inter.values())))
    fig, ax = plt.subplots(figsize=(8, 3.2)); w = 0.8 / len(inter)
    for k, (m, r) in enumerate(inter.items()):
        ax.bar(np.arange(len(names)) + (k - (len(inter) - 1) / 2) * w, [r[n]["err"] for n in names], w, label=m)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=25, fontsize=7); ax.legend(fontsize=6)
    ax.set_ylabel("доля неверных рёбер"); ax.set_title("ошибка прогноза графа при интервенциях в данные (тест)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{a.out}/fig_interventions.png", dpi=130)

    log("обобщение на неизвестные семейства / системы")
    lo_f = leave_out(ds, "family", params)
    json.dump({g: {m: J(r) for m, r in d.items()} for g, d in lo_f.items()}, open(f"{a.out}/leave_family.json", "w"), indent=1, ensure_ascii=False)
    viz.plot_methods({m: {g: lo_f[g][m] for g in lo_f} for m in next(iter(lo_f.values()))}, "exact",
                     "без семейства в обучении: доля точных структур", f"{a.out}/fig_leave_family.png")
    lo_s = leave_out(ds, "system", params)
    json.dump({g: {m: J(r) for m, r in d.items()} for g, d in lo_s.items()}, open(f"{a.out}/leave_system.json", "w"), indent=1, ensure_ascii=False)
    viz.plot_methods({m: {g: lo_s[g][m] for g in lo_s} for m in next(iter(lo_s.values()))}, "exact",
                     "без системы в обучении: доля точных структур", f"{a.out}/fig_leave_system.png")

    log("реальные данные Hudson Bay")
    hb = Prepared(hudson_bay())
    full = LinearGraphForecaster(**params).fit(ds.train_preps(list(range(len(ds.samples)))), dagger_rounds=1)
    no_lotka = [i for i, s in enumerate(ds.samples) if s.family != "lotka"]
    blind = LinearGraphForecaster(**params).fit(ds.train_preps(no_lotka), dagger_rounds=1)
    rep = {}
    for tag, m in (("обучено на всей коллекции", full), ("без семейства Лотки–Вольтерры", blind)):
        fc = m.forecast(hb)
        rep[tag] = dict(formula=fc.formula(), exact=bool((fc.Z == hb.s.Z).all()), steps=len(fc.steps))
        log(tag, rep[tag])
        key = "full" if m is full else "blind"
        viz.plot_sequence(fc, fname=f"{a.out}/fig_hudson_sequence_{key}.png")
        viz.plot_fit(fc, fname=f"{a.out}/fig_hudson_fit_{key}.png")
        viz.plot_sequence_metrics(fc, fname=f"{a.out}/fig_hudson_metrics_{key}.png")
        viz.plot_gamma(fc, fname=f"{a.out}/fig_hudson_gamma_{key}.png")
    for k, b in base.items():
        if b is not None:
            rep[k] = dict(exact=bool((b.predict(hb) == hb.s.Z).all()))
    json.dump(rep, open(f"{a.out}/hudson_bay.json", "w"), indent=1, ensure_ascii=False)

    log("примеры других систем из теста")
    seen = set()
    for i in te:
        s = ds.samples[i]
        if s.system in seen or s.system in ("harmonic", "damped"):
            continue
        seen.add(s.system)
        fc = model.forecast(ds.preps[i])
        n_show = 80 if s.system in ("lorenz", "rossler") else None
        viz.plot_sequence(fc, fname=f"{a.out}/fig_seq_{s.system}.png")
        viz.plot_fit(fc, polish=s.system not in ("lorenz", "rossler"), n_show=n_show, fname=f"{a.out}/fig_fit_{s.system}.png")
        plt.close("all")
    log("готово")


if __name__ == "__main__":
    main()
