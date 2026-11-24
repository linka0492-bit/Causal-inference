#!/usr/bin/env python
"""Надёжность ГА: несколько независимых запусков (разные seed). Для каждого – за сколько поколений
лучшая модель впервые стала эквивалентной Лотке–Вольтерре и совпал ли итоговый выбор по J.

    python run_seeds.py --seeds 0 1 2 3 4 --pop 80 --gen 40 --out results_seeds
"""
import argparse, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lab3.data import load_hudson_bay, make_scenarios
from lab3.ga import GAConfig, GeneticSR
from lab3.analysis import matches_lv, expanded_str, class_representatives


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--out", default="results_seeds")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ds = load_hudson_bay()
    rows, curves = [], {}
    for sd in a.seeds:
        ga = GeneticSR(ds, make_scenarios(ds), GAConfig(pop_size=a.pop, n_gen=a.gen, seed=sd), verbose=False)
        res = ga.run()
        first = next((g for g, sc in enumerate(res.chain) if matches_lv(sc.model)), None)
        reps, _ = class_representatives(res, ds)
        sel = reps[0]
        rows.append({"seed": sd, "first_lv_gen": first, "selected_is_lv": matches_lv(sel.model),
                     "J": sel.J, "n_archive": len(res.archive), "time": res.runtime,
                     "selected": expanded_str(sel.model, 3)})
        curves[sd] = [h["best_J"] for h in res.history]
        print(rows[-1], flush=True)
    with open(f"{a.out}/seeds.pkl", "wb") as f:
        pickle.dump({"rows": rows, "curves": curves}, f)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for sd, c in curves.items():
        ax[0].plot(c, label=f"seed {sd}")
    ax[0].set_xlabel("поколение"); ax[0].set_ylabel("лучшая J"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_ylim(min(min(c) for c in curves.values()) - 0.2, 0.5)
    ax[0].set_title("Сходимость при разных начальных seed")
    fg = [r["first_lv_gen"] if r["first_lv_gen"] is not None else a.gen for r in rows]
    ax[1].bar([str(r["seed"]) for r in rows], fg, color=["#2f9e44" if r["selected_is_lv"] else "#e03131" for r in rows])
    ax[1].set_xlabel("seed"); ax[1].set_ylabel("поколение первого попадания в класс ЛВ")
    ax[1].set_title(f"Найдено ЛВ в {sum(r['selected_is_lv'] for r in rows)} из {len(rows)} запусков (зелёный – итоговый выбор = ЛВ)")
    ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{a.out}/seeds.png", dpi=130)
    with open(f"{a.out}/seeds.md", "w", encoding="utf-8") as f:
        f.write("| seed | поколение 1-го попадания в ЛВ | выбран ЛВ | J | архив | время, c | выбранная модель |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['seed']} | {r['first_lv_gen']} | {r['selected_is_lv']} | {r['J']:.3f} | {r['n_archive']} | {r['time']:.0f} | `{r['selected']}` |\n")


if __name__ == "__main__":
    main()
