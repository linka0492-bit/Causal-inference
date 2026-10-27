"""
Эксперимент с мультипликативной z:

    dx/dt = (alpha - beta*y) * x * z,   dy/dt = (-gamma + delta*x) * y * z,
    z(t) = max(0, 1 + a * s(t)),  s = шум N(0,1) | константа 1 | sin(2*pi*t/10)

a = 0 — контроль (z = 1, исходная модель). Две части:
  fixed    — КС по траекториям при фиксированных параметрах (без мини-батчей)
  minibatch — 100 батчей по 12 из 21 точек, параметры подгоняются на каждом батче (LM)
Запуск: python3 zmul_experiment.py fixed | minibatch
"""
import json
import sys
import numpy as np
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import ks_2samp, wasserstein_distance

from lotka_volterra_zmul_tree import compile_tree

F = compile_tree()

HARE = np.array([30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1,
                 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7])
LYNX = np.array([4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6])
T_OBS = np.arange(21.0)
T_GRID = np.arange(0, 20.0001, 0.1)
P_NAMES = ["alpha", "beta", "gamma", "delta", "x0", "y0"]
THETA_FULL = np.array([0.481, 0.0248, 0.926, 0.0275, 34.9, 3.86])  # отчёт, LM по полной выборке
AMPS = [0.0, 0.05, 0.1, 0.2, 0.4]
SCEN = {"z_noise": ("noise", 0), "z_const": ("const",), "z_sin": ("sin",)}
BATCH, NB = 12, 100


def make_z(spec, a):
    kind = spec[0]
    if kind == "none":
        return lambda t: 1.0
    if kind == "noise":
        vals = np.random.default_rng(spec[1]).normal(0, 1, 21)
        return lambda t: max(0.0, 1.0 + a * np.interp(t, T_OBS, vals))
    if kind == "const":
        return lambda t: 1.0 + a
    if kind == "sin":
        return lambda t: max(0.0, 1.0 + a * np.sin(2 * np.pi * t / 10))
    raise ValueError(kind)


def simulate(theta, zf, t_eval, method="LSODA", rtol=1e-6, **kw):
    alpha, beta, gamma, delta, x0, y0 = theta

    def rhs(t, s):
        dx, dy = F(s[0], s[1], zf(t), alpha, beta, gamma, delta)
        return [dx, dy]

    sol = solve_ivp(rhs, (0, t_eval[-1]), [x0, y0], t_eval=t_eval, method=method,
                    rtol=rtol, atol=rtol, **kw)
    if not sol.success or sol.y.shape[1] != len(t_eval):
        return None
    return sol.y


def holm(p, alpha=0.05):
    p = np.asarray(p)
    order = np.argsort(p)
    sig = np.zeros(len(p), dtype=int)
    for rank, i in enumerate(order):
        if p[i] < alpha / (len(p) - rank):
            sig[i] = 1
        else:
            break
    return sig


# ---------------------------------------------------------------- часть 1: fixed
def run_fixed():
    def traj(a, spec):
        return simulate(THETA_FULL, make_z(spec, a), T_GRID, method="RK45", rtol=1e-8, max_step=0.1)

    base = traj(0.0, ("none",))
    rows, trajs = [], {}
    for name, spec in SCEN.items():
        for a in AMPS:
            tr = traj(a, spec)
            trajs[f"{name}|{a}"] = tr
            for j, var in enumerate(["x (заяц)", "y (рысь)"]):
                r = ks_2samp(base[j], tr[j])
                rows.append(dict(scenario=name, a=a, var=var, D=float(r.statistic), p=float(r.pvalue)))
    for r, s in zip(rows, holm([r["p"] for r in rows])):
        r["indicator"] = int(s)

    ens = []
    for a in AMPS[1:]:
        flags = []
        for seed in range(1, 51):
            tr = traj(a, ("noise", seed))
            flags.append(holm([ks_2samp(base[j], tr[j]).pvalue for j in range(2)]))
        flags = np.array(flags)
        ens.append(dict(a=a, share_x=float(flags[:, 0].mean()), share_y=float(flags[:, 1].mean())))

    np.savez("zmul_fixed_traj.npz", base=base, **trajs)
    json.dump(dict(rows=rows, ensemble=ens), open("zmul_fixed_results.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"{'сценарий':10s}{'a':>6s}{'перем.':>10s}{'D':>7s}{'p':>10s}  инд.")
    for r in rows:
        print(f"{r['scenario']:10s}{r['a']:6.2f}{r['var']:>10s}{r['D']:7.3f}{r['p']:10.2g}  {r['indicator']}")
    print("\nАнсамбль z_noise (50 розыгрышей): доля значимых")
    for e in ens:
        print(f"  a={e['a']:.2f}  x: {e['share_x']:.2f}   y: {e['share_y']:.2f}")


# ------------------------------------------------------------- часть 2: minibatch
def fit_batch(idx, zf, p0=THETA_FULL, max_nfev=25):
    def resid(p):
        sim = simulate(p, zf, T_OBS)
        if sim is None or not np.all(np.isfinite(sim)):
            return np.full(2 * len(idx), 1e3)
        return np.concatenate([sim[0][idx] - HARE[idx], sim[1][idx] - LYNX[idx]])
    return least_squares(resid, p0, method="lm", max_nfev=max_nfev).x


def run_config(args):
    a, spec, seed, nb = args
    zf = make_z(spec, a)
    rng = np.random.default_rng(seed)
    out = np.zeros((nb, 6))
    for k in range(nb):
        idx = np.sort(rng.choice(21, BATCH, replace=False))
        out[k] = fit_batch(idx, zf)
    return out


def run_minibatch():
    jobs, keys = [(0.0, ("none",), 1000, NB), (0.0, ("none",), 2000, NB)], [("base", 0.0), ("control", 0.0)]
    s = 3000
    for name, spec in SCEN.items():
        for a in AMPS[1:]:
            s += 1
            jobs.append((a, spec, s, NB)); keys.append((name, a))
    with Pool(2) as pool:
        outs = pool.map(run_config, jobs, chunksize=1)
    res = dict(zip(keys, outs))
    np.savez("zmul_minibatch_samples.npz", **{f"{k[0]}|{k[1]}": v for k, v in res.items()})

    base = res[("base", 0.0)]
    sd = base.std(axis=0, ddof=1)
    rows = []
    for k, v in res.items():
        if k[0] == "base":
            continue
        for j, p in enumerate(P_NAMES):
            r = ks_2samp(base[:, j], v[:, j])
            rows.append(dict(scenario=k[0], a=k[1], param=p, D=float(r.statistic), p=float(r.pvalue),
                             w1_sigma=float(wasserstein_distance(base[:, j], v[:, j]) / sd[j]),
                             mean=float(v[:, j].mean()), std=float(v[:, j].std(ddof=1))))
    for r, s_ in zip(rows, holm([r["p"] for r in rows])):
        r["indicator"] = int(s_)
    base_stats = {p: dict(mean=float(base[:, j].mean()), std=float(sd[j])) for j, p in enumerate(P_NAMES)}
    json.dump(dict(rows=rows, base=base_stats, n_tests=len(rows)),
              open("zmul_minibatch_results.json", "w"), ensure_ascii=False, indent=1)
    print("base mean/std:", {p: (round(v['mean'], 4), round(v['std'], 4)) for p, v in base_stats.items()})
    print(f"\n{'сценарий':10s}{'a':>6s}  " + "  ".join(f"{p:>14s}" for p in P_NAMES))
    for name, a in [k for k in keys if k[0] != "base"]:
        sel = [r for r in rows if r["scenario"] == name and r["a"] == a]
        print(f"{name:10s}{a:6.2f}  " + "  ".join(
            f"{r['indicator']} D={r['D']:.2f} W={r['w1_sigma']:.1f}".rjust(14) for r in sel))


if __name__ == "__main__":
    {"fixed": run_fixed, "minibatch": run_minibatch}[sys.argv[1]]()
