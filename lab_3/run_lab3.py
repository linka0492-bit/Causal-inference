#!/usr/bin/env python
"""Полный запуск лабораторной работы № 3: ГА символьной регрессии для Лотки–Вольтерры.

    python run_lab3.py --out results --seed 0 --pop 120 --gen 60

Создаёт в папке --out: figures/*.png, archive.csv (выборка графов), archive.pkl, result.pkl, summary.md
"""
import argparse, os, pickle, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab3.data import (load_hudson_bay, make_scenarios, zero_lynx, noise, noise_cond, shuffle)
from lab3.expr import true_lv_model
from lab3.fit import IntegralProblem, fit_trajectory
from lab3.ga import GAConfig, GeneticSR
from lab3.objective import Weights, score
from lab3.interventions import StabilityEvaluator, intervention_table
from lab3.analysis import (class_representatives, canonical_lv, refine_top, chain_trajectories, trajectory_info, expanded_str, matches_lv,
                           lv_coefficients)
from lab3 import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=120)
    ap.add_argument("--gen", type=int, default=60)
    ap.add_argument("--w_cmp", type=float, default=0.03)
    ap.add_argument("--w_stab", type=float, default=1.0)
    ap.add_argument("--reuse", action="store_true", help="загрузить сохранённый результат ГА из --out/ga_result.pkl")
    ap.add_argument("--skip_table", action="store_true", help="не считать таблицу интервенций (долго)")
    a = ap.parse_args()
    fig_dir = os.path.join(a.out, "figures"); os.makedirs(fig_dir, exist_ok=True)

    ds = load_hudson_bay()
    w = Weights(w_cmp=a.w_cmp, w_stab=a.w_stab)
    cfg = GAConfig(pop_size=a.pop, n_gen=a.gen, seed=a.seed, weights=w)
    print("== Запуск ГА:", cfg.pop_size, "особей,", cfg.n_gen, "поколений, seed", cfg.seed, flush=True)
    pk = os.path.join(a.out, "ga_result.pkl")
    if a.reuse and os.path.exists(pk):
        with open(pk, "rb") as f:
            res = pickle.load(f)
        cfg = res.config
        print("Загружен сохранённый результат ГА", flush=True)
    else:
        ga = GeneticSR(ds, make_scenarios(ds), cfg)
        res = ga.run()
        with open(pk, "wb") as f:
            pickle.dump(res, f)
    print(f"ГА завершён за {res.runtime:.0f} c, оценено {res.n_evaluated} моделей, "
          f"уникальных структур в архиве: {len(res.archive)}", flush=True)

    # --- эталон ЛВ в той же целевой функции
    true_m = true_lv_model()
    prob = IntegralProblem(ds)
    stab = StabilityEvaluator(make_scenarios(ds))
    true_sc = score(true_m, prob, stab, w, restarts=2, rng=np.random.default_rng(0))
    print("Эталон ЛВ:", true_sc.summary())

    # --- траектории цепочки и уточнение топа
    traj = chain_trajectories(res.chain, ds)
    print("Считаю траектории для цепочки порождённых графов…", flush=True)
    traj_s = chain_trajectories(res.sampled_chain, ds, max_nfev=15)
    print("Группирую граф по классам эквивалентности (раскрытие скобок)…", flush=True)
    reps, counts = class_representatives(res, ds)
    n_lv = sum(1 for sc in res.archive.values() if sc.acc < 0.35 and matches_lv(sc.model))
    print(f"Синтаксически разных графов: {len(res.archive)}; классов эквивалентности: {len(reps)}; графов, эквивалентных ЛВ: {n_lv}", flush=True)
    rows = refine_top(res, ds, topk=10, reps=reps)
    # ВЫБОР графа – по целевой функции J (интегральная подгонка); уточнение по траектории – проверка
    rows_J = sorted(rows, key=lambda r: r["scored"].J)
    final = rows_J[0]
    best_model, info = final["info"]["model"], final["info"]
    print("\nВыбранный граф (после уточнения по траектории):", best_model)
    print("Раскрыто:", expanded_str(best_model), "| эквивалент ЛВ:", final["lv_equiv"], "| RSS =", round(final["rss"], 1))

    # --- рисунки
    viz.fig_convergence(res, f"{fig_dir}/01_convergence.png")
    viz.fig_chain_metrics(res.chain, traj, f"{fig_dir}/02_chain_metrics.png")
    viz.fig_chain_pages(res.chain, traj, ds, path_prefix=f"{fig_dir}/03_chain_best_page", title="Лучшая (запрунённая) модель по поколениям (только шаги с изменением)")
    viz.fig_chain_metrics(res.sampled_chain, traj_s, f"{fig_dir}/02b_sampled_chain_metrics.png", title="Лучший ПОРОЖДЁННЫЙ граф каждого поколения (последовательность семплирования)")
    for f_ in viz.fig_chain_pages(res.sampled_chain, traj_s, ds, per_page=12, ncols=4, dedupe=False, path_prefix=f"{fig_dir}/03b_sampled_chain_page", title="Последовательность порождаемых графов (лучший потомок поколения)"):
        plt.close(f_)
    viz.fig_pareto(res, true_sc, f"{fig_dir}/04_pareto.png")
    viz.fig_fit(info, best_model, ds, path=f"{fig_dir}/05_fit.png")
    viz.fig_objective_terms(res, 12, f"{fig_dir}/06_objective_terms.png", arch=reps)
    viz.fig_weight_sensitivity(res, true_sc, f"{fig_dir}/07_weight_sensitivity.png", arch=reps)
    viz.fig_incidence(best_model, f"{fig_dir}/08_incidence.png")

    # --- do(u=1): найденная модель vs эталон vs конкуренты
    true_info = trajectory_info(true_m, ds)
    models = {"найденный граф": (info["model"], info["x0"], info["y0"], dict(c="C3", lw=2.2)),
              "эталон ЛВ (ЛР №1)": (true_info["model"], true_info["x0"], true_info["y0"], dict(c="k", lw=1.2, ls="--"))}
    comp = [sc for sc in reps if not matches_lv(sc.model) and sc.acc < 0.35][:3]
    for k, sc in enumerate(comp):
        mi = trajectory_info(sc.model, ds)
        if np.isfinite(mi["rss"]):
            lab = expanded_str(sc.model, 2).replace("dx/dt", "x'").replace("dy/dt", "y'").replace("*", "")
            models[f"конкурент {k + 1} (RSS={mi['rss']:.0f}): {lab}"] = (mi["model"], mi["x0"], mi["y0"], dict(c=f"C{k}", lw=1.3))
    viz.fig_do_love(models, ds, beta=0.0248, path=f"{fig_dir}/09_do_love.png")

    # --- таблица интервенций в данные для выбранного графа
    if not a.skip_table:
        sc_best = res.archive[best_model.key()] if best_model.key() in res.archive else res.best
        from lab3.fit import fit_model
        mm, canon_labels = canonical_lv(best_model)   # параметры в раскрытой форме (α, β, δ, γ), как в ЛР № 2
        fit_model(mm, prob)   # интегральные параметры для мини-батч процедуры
        scen = {"zero_lynx": zero_lynx(ds), "noise": noise(ds, 1), "noise_cond": noise_cond(ds, 1),
                "shuffle30": shuffle(ds, 1), "small_sample(k=5)": (ds, 5), "bias_+0.5": noise(ds, 1, bias=0.5)}
        print("Считаю таблицу интервенций (мини-батчи)…", flush=True)
        tab = intervention_table(mm, ds, scen, n_batches=60, k=12)
        labels = canon_labels or [f"θ{i + 1}" for i in range(tab["n_params"])]
        viz.fig_intervention_heatmaps(tab, labels, f"{fig_dir}/10_intervention_heatmaps.png")
        pick = {k: tab["samples"][k] for k in ["noise", "shuffle30", "bias_+0.5"]}
        viz.fig_param_hists(tab, pick, labels, f"{fig_dir}/11_param_hists.png")
        with open(f"{a.out}/intervention_table.pkl", "wb") as f:
            pickle.dump({k: v for k, v in tab.items()}, f)

    # --- сохранение выборки графов
    import csv
    with open(f"{a.out}/archive.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["rank", "J", "acc_1mR2", "complexity", "n_nodes", "n_params", "instability", "gen_first", "lv_equivalent", "model", "expanded"])
        for r, s in enumerate(sorted(res.archive.values(), key=lambda s: s.J)):
            wr.writerow([r + 1, f"{s.J:.4f}", f"{s.acc:.5f}", f"{s.complexity:.1f}", s.n_nodes, s.n_params,
                         f"{s.instab:.5f}", s.gen, matches_lv(s.model), str(s.model), expanded_str(s.model)])
    with open(f"{a.out}/result.pkl", "wb") as f:
        pickle.dump({"history": res.history, "chain": res.chain, "archive": res.archive, "cfg": cfg,
                     "final": best_model, "final_rss": final["rss"]}, f)

    with open(f"{a.out}/summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Итоги запуска (seed={a.seed}, pop={a.pop}, gen={a.gen})\n\n")
        f.write(f"- время: {res.runtime:.0f} c; оценено моделей: {res.n_evaluated}; архив: {len(res.archive)} структур\n")
        f.write(f"- выбранный граф: `{best_model}`\n- раскрыто: `{expanded_str(best_model)}`\n")
        f.write(f"- эквивалент Лотке–Вольтерре: {final['lv_equiv']}; RSS по траектории: {final['rss']:.1f} (эталон из ЛР №2: 595)\n")
        f.write(f"- эталон ЛВ в той же J: {true_sc.J:.3f} (найденный: {final['scored'].J:.3f})\n\n## Топ-10 классов эквивалентности\n\n")
        f.write("(выбор графа – по J; ниже для сравнения и ранжирование по J_final с точностью по траектории)\n\n")
        for r in rows_J:
            f.write(f"- J={r['scored'].J:.3f} ")
            f.write(f"J_final={r['J_final']:.3f} RSS={r['rss']:.1f} ЛВ={r['lv_equiv']}: {r['expanded']}\n")
    print("Готово. Результаты в", a.out)


if __name__ == "__main__":
    main()
