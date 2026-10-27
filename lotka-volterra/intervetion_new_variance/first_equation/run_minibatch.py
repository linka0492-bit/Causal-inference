"""Запуск мини-батч эксперимента с z: base, контроль, 3 варианта z x 4 значения lam."""
import json
import sys
import numpy as np
from multiprocessing import Pool
from scipy.stats import ks_2samp, wasserstein_distance

import minibatch_z as mb

NB = 100
LAMS = [0.01, 0.02, 0.05, 0.1]
SCEN = {"z_noise": ("noise", 0), "z_const": ("const",), "z_sin": ("sin",)}


def main():
    jobs, keys = [], []
    jobs.append((0.0, ("none",), 1000, NB)); keys.append(("base", 0.0))
    jobs.append((0.0, ("none",), 2000, NB)); keys.append(("control", 0.0))
    s = 3000
    for name, spec in SCEN.items():
        for lam in LAMS:
            s += 1
            jobs.append((lam, spec, s, NB)); keys.append((name, lam))
    with Pool(2) as pool:
        outs = pool.map(mb.run_config, jobs, chunksize=1)
    res = dict(zip(keys, outs))
    np.savez("minibatch_samples.npz", **{f"{k[0]}|{k[1]}": v for k, v in res.items()})

    base = res[("base", 0.0)]
    sd = base.std(axis=0, ddof=1)
    rows = []
    for k, v in res.items():
        if k[0] == "base":
            continue
        for j, p in enumerate(mb.P_NAMES):
            r = ks_2samp(base[:, j], v[:, j])
            rows.append(dict(scenario=k[0], lam=k[1], param=p, D=float(r.statistic),
                             p=float(r.pvalue),
                             w1_sigma=float(wasserstein_distance(base[:, j], v[:, j]) / sd[j]),
                             mean=float(v[:, j].mean()), std=float(v[:, j].std(ddof=1))))
    sig = mb.holm([r["p"] for r in rows])
    for r, s_ in zip(rows, sig):
        r["indicator"] = int(s_)
    base_stats = {p: dict(mean=float(base[:, j].mean()), std=float(sd[j]))
                  for j, p in enumerate(mb.P_NAMES)}
    json.dump(dict(rows=rows, base=base_stats, n_tests=len(rows)),
              open("minibatch_results.json", "w"), ensure_ascii=False, indent=1)

    print("base mean/std:", {p: (round(v['mean'], 4), round(v['std'], 4)) for p, v in base_stats.items()})
    print(f"\n{'сценарий':10s}{'lam':>6s}  " + "  ".join(f"{p:>12s}" for p in mb.P_NAMES))
    for name, lam in [k for k in keys if k[0] != "base"]:
        sel = [r for r in rows if r["scenario"] == name and r["lam"] == lam]
        print(f"{name:10s}{lam:6.3f}  " + "  ".join(
            f"{r['indicator']} D={r['D']:.2f} W={r['w1_sigma']:.1f}".rjust(12) for r in sel))


if __name__ == "__main__":
    main()
