"""Интервенции в данные (как в ЛР № 1–3). Каждая принимает Sample и rng, возвращает новый Sample.

Структура модели (Z) при интервенции в данные не меняется: она остаётся эталоном, а вопрос
в том, насколько устойчив к порче данных прогноз структуры.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator


def _states(s):
    return slice(0, s.n_eq)


def noise(s, rng, level=0.3):
    X = s.X.copy()
    sd = X[:, _states(s)].std(axis=0)
    X[:, _states(s)] += level * sd * rng.standard_normal((len(X), s.n_eq))
    return s.replace(X=X, tag="noise")


def noise_cond(s, rng, level=0.3):
    """Шум первой переменной пропорционален значению второй (для одной переменной — своей)."""
    X = s.X.copy()
    sd = X[:, _states(s)].std(axis=0)
    drv = X[:, 1] if s.n_eq > 1 else X[:, 0]
    scale = np.abs(drv) / (np.abs(drv).mean() + 1e-12)
    X[:, 0] += level * sd[0] * scale * rng.standard_normal(len(X))
    if s.n_eq > 1:
        X[:, 1:s.n_eq] += level * sd[1:] * rng.standard_normal((len(X), s.n_eq - 1))
    return s.replace(X=X, tag="noise_cond")


def small_sample(s, rng, frac=12 / 21):
    m = len(s.t)
    k = max(int(round(frac * m)), 8)
    idx = np.sort(rng.choice(m, size=k, replace=False))
    return s.replace(t=s.t[idx], X=s.X[idx], tag="small_sample")


def bias(s, rng, level=0.25, shift=0.25):
    X = s.X.copy()
    sd = X[:, _states(s)].std(axis=0)
    X[:, _states(s)] += (shift + level * rng.standard_normal((len(X), s.n_eq))) * sd
    return s.replace(X=X, tag="bias")


def zero_second(s, rng):
    """Вторая переменная заменена константой (среднее)."""
    X = s.X.copy()
    j = 1 if s.n_eq > 1 else 0
    X[:, j] = X[:, j].mean()
    return s.replace(X=X, tag="zero_second")


def shuffle30(s, rng, frac=0.3):
    """У 30 % моментов времени первая и вторая переменные поменялись местами."""
    X = s.X.copy()
    if s.n_eq > 1:
        idx = rng.choice(len(X), size=int(frac * len(X)), replace=False)
        X[idx, 0], X[idx, 1] = s.X[idx, 1], s.X[idx, 0]
    return s.replace(X=X, tag="shuffle30")


def add_points(s, rng, n_new=5, factor=1.0):
    """Добавить n_new точек в середины интервалов (значения — PCHIP-интерполяция + шум).
    factor != 1 умножает первую переменную добавленных точек (аналог add_shift из ЛР № 1)."""
    mid = np.sort(rng.choice(len(s.t) - 1, size=min(n_new, len(s.t) - 1), replace=False))
    tn = 0.5 * (s.t[mid] + s.t[mid + 1])
    Xn = np.column_stack([PchipInterpolator(s.t, s.X[:, v])(tn) for v in range(s.n_vars)])
    sd = s.X[:, :s.n_eq].std(axis=0)
    Xn[:, :s.n_eq] += 0.05 * sd * rng.standard_normal((len(tn), s.n_eq))
    Xn[:, 0] *= factor
    t = np.concatenate([s.t, tn])
    X = np.vstack([s.X, Xn])
    o = np.argsort(t)
    return s.replace(t=t[o], X=X[o], tag="add_same" if factor == 1.0 else "add_shift")


INTERVENTIONS = dict(noise=noise, noise_cond=noise_cond, small_sample=small_sample, bias=bias,
                     zero_second=zero_second, shuffle30=shuffle30, add_points=add_points)

# набор из ЛР № 3 для оценки устойчивости: 4 вида порчи × 2 розыгрыша
STABILITY_SET = ("noise", "noise_cond", "small_sample", "bias")


def corrupt(s, name, rng):
    return INTERVENTIONS[name](s, rng)


def stability_copies(s, rng, draws=2):
    return [corrupt(s, n, rng) for n in STABILITY_SET for _ in range(draws)]
