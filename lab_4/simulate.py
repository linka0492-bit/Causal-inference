"""Решение ОДУ по спрогнозированной модели и «честная» подгонка параметров по траектории
(как в ЛР № 3: структура из интегрального согласования, числа уточняются по решению ОДУ)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares


def _rhs(sample, Z, C):
    terms = [np.array(e) for e in sample.terms]
    n_eq, n_vars = sample.n_eq, sample.n_vars
    act = [[(terms[j], C[i, j]) for j in np.flatnonzero(Z[i])] for i in range(n_eq)]
    drv = [PchipInterpolator(sample.t, sample.X[:, v]) for v in range(n_eq, n_vars)]

    def f(t, s):
        v = np.concatenate([s, [d(t) for d in drv]]) if drv else s
        return np.array([sum(c * np.prod(v ** e) for e, c in eq) for eq in act])
    return f


def simulate(sample, Z, C, x0=None, t=None):
    """Траектория модели (len(t), n_eq); NaN, если решение расходится."""
    t = sample.t if t is None else t
    x0 = sample.X[0, :sample.n_eq] if x0 is None else x0
    try:
        sol = solve_ivp(_rhs(sample, Z, C), [t[0], t[-1]], x0, t_eval=t, method="LSODA",
                        rtol=1e-6, atol=1e-8)
        if sol.y.shape[1] == len(t) and np.all(np.isfinite(sol.y)):
            return sol.y.T
    except Exception:
        pass
    return np.full((len(t), sample.n_eq), np.nan)


def trajectory_rss(sample, Z, C, x0=None):
    Y = simulate(sample, Z, C, x0)
    if np.isnan(Y).any():
        return np.inf
    return float(((Y - sample.X[:, :sample.n_eq]) ** 2).sum())


def polish_params(sample, Z, C0, max_nfev=200):
    """Уточнить коэффициенты активных рёбер и начальные условия по траектории (МНК, LM/TRF)."""
    idx = np.argwhere(Z)
    n_eq = sample.n_eq
    sd = sample.X[:, :n_eq].std(axis=0) + 1e-12
    obs = sample.X[:, :n_eq]
    th0 = np.concatenate([C0[Z], obs[0]])

    def unpack(th):
        C = np.zeros_like(C0)
        C[Z] = th[:len(idx)]
        return C, th[len(idx):]

    def res(th):
        C, x0 = unpack(th)
        Y = simulate(sample, Z, C, x0)
        if np.isnan(Y).any():
            return np.full(obs.size, 1e3)
        return ((Y - obs) / sd).ravel()

    out = least_squares(res, th0, method="trf", max_nfev=max_nfev, x_scale=np.maximum(np.abs(th0), 1e-3))
    C, x0 = unpack(out.x)
    return C, x0, trajectory_rss(sample, Z, C, x0)
