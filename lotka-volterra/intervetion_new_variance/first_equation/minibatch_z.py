"""
Мини-батч процедура (A из отчёта) для модели Лотки-Вольтерры с переменной z.

Батч = случайные 12 из 21 моментов наблюдения. На каждом батче заново
подгоняются theta = (alpha, beta, gamma, delta, x0, y0) методом
Левенберга-Марквардта (25 вызовов функции невязки на итерацию-лимит как в
отчёте: max_nfev=25). Параметр lam фиксирован, z задана функцией времени.
Правая часть берётся из дерева (lotka_volterra_z_tree), скомпилированного
в python-функцию (без sympy).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import ks_2samp, wasserstein_distance

from lotka_volterra_tree import NodeColor
from lotka_volterra_z_tree import build_lotka_volterra_z

HARE = np.array([30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1,
                 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7])
LYNX = np.array([4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
                 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6])
T_OBS = np.arange(21.0)
P_NAMES = ["alpha", "beta", "gamma", "delta", "x0", "y0"]
THETA_FULL = np.array([0.481, 0.0248, 0.926, 0.0275, 34.9, 3.86])  # отчёт, LM по полной выборке
BATCH = 12


def compile_tree():
    """Генерирует python-функцию f(x,y,z,lam,alpha,beta,gamma,delta)->(dx,dy) из DAG."""
    b = build_lotka_volterra_z()
    tree = b["tree"]
    lines = []
    for n in tree.nodes:
        if not n.parents:
            lines.append(f"    v{n.id} = {n.name}")
        else:
            a = [f"v{p.id}" for p in n.parents]
            if n.op == "+":
                e = " + ".join(a)
            elif n.op == "-":
                e = a[0] + "".join(f" - {t}" for t in a[1:])
            elif n.op == "*":
                e = " * ".join(a)
            else:
                e = f"{a[0]} / {a[1]}"
            lines.append(f"    v{n.id} = {e}")
    src = ("def f(x, y, z, lam, alpha, beta, gamma, delta):\n" + "\n".join(lines) +
           f"\n    return v{b['dxdt_root'].id}, v{b['dydt_root'].id}\n")
    ns = {}
    exec(src, ns)
    return ns["f"], src


F, F_SRC = compile_tree()


def simulate(theta, lam, zf, t_eval=T_OBS):
    alpha, beta, gamma, delta, x0, y0 = theta

    def rhs(t, s):
        dx, dy = F(s[0], s[1], zf(t), lam, alpha, beta, gamma, delta)
        return [dx, dy]

    sol = solve_ivp(rhs, (0, t_eval[-1]), [x0, y0], t_eval=t_eval, method="LSODA",
                    rtol=1e-6, atol=1e-6)
    if not sol.success or sol.y.shape[1] != len(t_eval):
        return None
    return sol.y


def fit_batch(idx, lam, zf, p0=THETA_FULL, max_nfev=25):
    tb = T_OBS[idx]

    def resid(p):
        # t_eval должен содержать 0: считаем на всех целых годах, берём нужные
        sim = simulate(p, lam, zf)
        if sim is None or not np.all(np.isfinite(sim)):
            return np.full(2 * len(idx), 1e3)
        return np.concatenate([sim[0][idx] - HARE[idx], sim[1][idx] - LYNX[idx]])

    r = least_squares(resid, p0, method="lm", max_nfev=max_nfev)
    return r.x


def run_config(args):
    """args = (lam, z_spec, seed, n_batches); z_spec строится внутри процесса."""
    lam, z_spec, seed, nb = args
    zf = make_z(z_spec)
    rng = np.random.default_rng(seed)
    out = np.zeros((nb, 6))
    for k in range(nb):
        idx = np.sort(rng.choice(21, BATCH, replace=False))
        out[k] = fit_batch(idx, lam, zf)
    return out


def make_z(spec):
    kind = spec[0]
    if kind == "none":
        return lambda t: 0.0
    if kind == "noise":
        vals = np.random.default_rng(spec[1]).normal(0, 1, 21)
        return lambda t: np.interp(t, T_OBS, vals)
    if kind == "const":
        return lambda t: 1.0
    if kind == "sin":
        return lambda t: np.sin(2 * np.pi * t / 10)
    raise ValueError(kind)


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


if __name__ == "__main__":
    import time
    t = time.time()
    r = run_config((0.05, ("sin",), 1, 4))
    print("4 батча:", time.time() - t, "c")
    print(r)
