"""Коллекция «(выборка, целевая модель)».

Каждая система задана своими истинными слагаемыми: список словарей {показатели степеней: коэффициент}
по одному на уравнение. Из них одновременно получаются (а) правая часть ОДУ для генерации
данных и (б) истинная структура Z (ground truth), поэтому они не могут разойтись.

Чтобы добавить систему, напишите функцию rng -> dict(terms=..., x0=..., t=..., driver=...)
и зарегистрируйте её через register_system().
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .library import make_library


# ---------------------------------------------------------------- выборка
@dataclass
class Sample:
    system: str
    family: str
    t: np.ndarray
    X: np.ndarray                  # (m, n_vars): состояния, затем внешние переменные (драйверы)
    names: tuple
    n_eq: int                      # число уравнений (первые n_eq столбцов X)
    Z: np.ndarray                  # bool (n_eq, n_terms): истинная структура
    W: np.ndarray                  # float (n_eq, n_terms): истинные коэффициенты (nan — неизвестны)
    tag: str = "clean"             # какая интервенция применена к данным
    real: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def n_vars(self):
        return self.X.shape[1]

    @property
    def terms(self):
        return make_library(self.n_vars)

    def replace(self, **kw):
        d = dict(self.__dict__)
        d.update(kw)
        return Sample(**d)


# ---------------------------------------------------------------- генерация данных ОДУ
def _driver(rng, kind=None, amp=1.0, T=20.0):
    """Внешняя переменная z(t): синусоида или шум (раз в год своё значение, между ними линейно)."""
    kind = kind or rng.choice(["sin", "noise"])
    if kind == "sin":
        per, ph = rng.uniform(7, 13), rng.uniform(0, 2 * np.pi)
        return lambda t: amp * np.sin(2 * np.pi * np.asarray(t) / per + ph)
    knots = np.arange(0, int(T) + 2)
    vals = amp * rng.standard_normal(len(knots))
    return lambda t: np.interp(t, knots, vals)


def _rhs_factory(terms, n_vars, driver):
    spec = [[(np.array(e), c) for e, c in eq.items()] for eq in terms]
    n_eq = len(terms)

    def rhs(t, s):
        v = np.empty(n_vars)
        v[:n_eq] = s
        if n_vars > n_eq:
            v[n_eq:] = driver(t)
        out = np.empty(n_eq)
        for i, eq in enumerate(spec):
            out[i] = sum(c * np.prod(v ** e) for e, c in eq)
        return out
    return rhs


def _lotka_params(rng):
    j = lambda x, r=0.25: x * rng.uniform(1 - r, 1 + r)
    return j(0.481), j(0.0248), j(0.926), j(0.0275)


def _lotka_x0(rng):
    return np.array([34.9 * rng.uniform(0.7, 1.3), 3.86 * rng.uniform(0.7, 1.5)])


def _grid(rng, T=20.0, choices=(21, 41, 81), p=(0.5, 0.3, 0.2)):
    return np.linspace(0, T, int(rng.choice(choices, p=p)))


def lotka(rng):
    a, b, g, d = _lotka_params(rng)
    return dict(terms=[{(1, 0): a, (1, 1): -b}, {(1, 1): d, (0, 1): -g}],
                x0=_lotka_x0(rng), t=_grid(rng), driver=None)


def lotka_logistic(rng):
    a, b, g, d = _lotka_params(rng)
    c = a / rng.uniform(120, 200)
    return dict(terms=[{(1, 0): a, (2, 0): -c, (1, 1): -b}, {(1, 1): d, (0, 1): -g}],
                x0=_lotka_x0(rng), t=_grid(rng), driver=None)


def lotka_z_in(rng):
    """ЛР № 2, способ 1: z внутри скобок, dx = (a - b y + lam z) x, dy = (-g + d x + lam z) y."""
    a, b, g, d = _lotka_params(rng)
    lam = rng.uniform(0.04, 0.12)
    return dict(terms=[{(1, 0, 0): a, (1, 1, 0): -b, (1, 0, 1): lam},
                       {(1, 1, 0): d, (0, 1, 0): -g, (0, 1, 1): lam}],
                x0=_lotka_x0(rng), t=_grid(rng), driver=_driver(rng))


def lotka_z_mult(rng):
    """ЛР № 2, способ 2: множитель z = 1 + a s(t) перед правой частью."""
    a, b, g, d = _lotka_params(rng)
    amp = rng.uniform(0.15, 0.4)
    s = _driver(rng, amp=amp)
    return dict(terms=[{(1, 0, 1): a, (1, 1, 1): -b}, {(1, 1, 1): d, (0, 1, 1): -g}],
                x0=_lotka_x0(rng), t=_grid(rng), driver=lambda t: 1.0 + s(t))


def lotka_z_add(rng):
    """ЛР № 2, способ 3: аддитивный источник eta*z, theta*z."""
    a, b, g, d = _lotka_params(rng)
    eta, theta = rng.uniform(0.3, 1.0, size=2)
    return dict(terms=[{(1, 0, 0): a, (1, 1, 0): -b, (0, 0, 1): eta},
                       {(1, 1, 0): d, (0, 1, 0): -g, (0, 0, 1): theta}],
                x0=_lotka_x0(rng), t=_grid(rng), driver=_driver(rng))


def van_der_pol(rng):
    mu, om = rng.uniform(0.5, 2.0), rng.uniform(0.8, 1.3)
    return dict(terms=[{(0, 1): 1.0}, {(0, 1): mu, (2, 1): -mu, (1, 0): -om ** 2}],
                x0=np.array([rng.uniform(1.5, 2.5), rng.uniform(-0.3, 0.3)]),
                t=np.linspace(0, 20, int(rng.choice([60, 120]))), driver=None)


def harmonic(rng):
    """«Взаимная любовь» из ЛР № 1: dx = y, dy = -x (с разбросом частот)."""
    a, b = rng.uniform(0.7, 1.5, size=2)
    return dict(terms=[{(0, 1): a}, {(1, 0): -b}],
                x0=np.array([rng.uniform(0.5, 2.0), rng.uniform(-1.0, 1.0)]),
                t=np.linspace(0, 20, 50), driver=None)


def damped(rng):
    k, c = rng.uniform(0.5, 2.0), rng.uniform(0.1, 0.5)
    return dict(terms=[{(0, 1): 1.0}, {(1, 0): -k, (0, 1): -c}],
                x0=np.array([rng.uniform(1.0, 3.0), rng.uniform(-1.0, 1.0)]),
                t=np.linspace(0, 20, 60), driver=None)


def cooling(rng):
    k, tenv = rng.uniform(0.1, 0.4), rng.uniform(10, 25)
    return dict(terms=[{(1,): -k, (0,): k * tenv}], x0=np.array([rng.uniform(60, 100)]),
                t=np.linspace(0, 20, 30), driver=None)


def lorenz(rng):
    s, r, b = rng.uniform(9, 11), rng.uniform(26, 30), rng.uniform(2.4, 2.9)
    return dict(terms=[{(1, 0, 0): -s, (0, 1, 0): s},
                       {(1, 0, 0): r, (0, 1, 0): -1.0, (1, 0, 1): -1.0},
                       {(1, 1, 0): 1.0, (0, 0, 1): -b}],
                x0=rng.uniform(-8, 8, size=3) + np.array([0, 0, 20]),
                t=np.linspace(0, 4, 201), driver=None)


def rossler(rng):
    a, b, c = rng.uniform(0.15, 0.25), rng.uniform(0.15, 0.25), rng.uniform(5.0, 6.0)
    return dict(terms=[{(0, 1, 0): -1.0, (0, 0, 1): -1.0},
                       {(1, 0, 0): 1.0, (0, 1, 0): a},
                       {(0, 0, 0): b, (1, 0, 1): 1.0, (0, 0, 1): -c}],
                x0=np.array([1.0, 1.0, 0.0]) + 0.3 * rng.standard_normal(3),
                t=np.linspace(0, 40, 201), driver=None)


# имя -> (семейство, имена переменных, число уравнений, функция, максимальный относительный шум)
SYSTEMS = {
    "lotka":          ("lotka", ("x", "y"), 2, lotka, 0.08),
    "lotka_logistic": ("lotka", ("x", "y"), 2, lotka_logistic, 0.08),
    "lotka_z_in":     ("lotka", ("x", "y", "z"), 2, lotka_z_in, 0.08),
    "lotka_z_mult":   ("lotka", ("x", "y", "z"), 2, lotka_z_mult, 0.08),
    "lotka_z_add":    ("lotka", ("x", "y", "z"), 2, lotka_z_add, 0.08),
    "van_der_pol":    ("oscillator", ("x", "v"), 2, van_der_pol, 0.08),
    "harmonic":       ("oscillator", ("x", "y"), 2, harmonic, 0.08),
    "damped":         ("oscillator", ("x", "v"), 2, damped, 0.08),
    "cooling":        ("cooling", ("T",), 1, cooling, 0.08),
    "lorenz":         ("chaos", ("x", "y", "z"), 3, lorenz, 0.02),
    "rossler":        ("chaos", ("x", "y", "z"), 3, rossler, 0.03),
}


def register_system(name, family, names, n_eq, fn, noise_max=0.08):
    """Добавить свою систему в коллекцию (fn: rng -> dict(terms, x0, t, driver))."""
    SYSTEMS[name] = (family, tuple(names), n_eq, fn, noise_max)


def _truth(terms, n_vars):
    lib = make_library(n_vars)
    index = {e: j for j, e in enumerate(lib)}
    Z = np.zeros((len(terms), len(lib)), bool)
    W = np.zeros((len(terms), len(lib)))
    for i, eq in enumerate(terms):
        for e, c in eq.items():
            Z[i, index[tuple(e)]] = True
            W[i, index[tuple(e)]] = c
    return Z, W


def make_instance(name, rng, noise=None, max_tries=30):
    """Один экземпляр системы: случайные параметры, начальные условия, шум."""
    family, names, n_eq, fn, noise_max = SYSTEMS[name]
    n_vars = len(names)
    for _ in range(max_tries):
        spec = fn(rng)
        driver = spec["driver"]
        rhs = _rhs_factory(spec["terms"], n_vars, driver)
        t = spec["t"]
        sol = solve_ivp(rhs, [t[0], t[-1]], spec["x0"], t_eval=t, rtol=1e-9, atol=1e-9,
                        max_step=0.25 if driver is not None else np.inf)
        if not sol.success or sol.y.shape[1] != len(t):
            continue
        S = sol.y.T
        if not np.all(np.isfinite(S)) or np.abs(S).max() > 1e4:
            continue
        sigma = rng.uniform(0.0, noise_max) if noise is None else noise
        S = S + sigma * S.std(axis=0) * rng.standard_normal(S.shape)
        X = S if driver is None else np.column_stack([S, np.atleast_1d(driver(t))])
        Z, W = _truth(spec["terms"], n_vars)
        return Sample(name, family, t, X, names, n_eq, Z, W, meta=dict(noise=float(sigma)))
    raise RuntimeError(f"не удалось сгенерировать экземпляр {name}")


def build_collection(n_per_system=30, seed=0, systems=None):
    """Коллекция пар (выборка, модель): n_per_system экземпляров каждой системы."""
    rng = np.random.default_rng(seed)
    out = []
    for name in (systems or SYSTEMS):
        out.extend(make_instance(name, rng) for _ in range(n_per_system))
    return out


# ---------------------------------------------------------------- реальные данные
HUDSON_BAY = np.array([  # тыс. шкур: заяц, рысь; 1900–1920
    [30.0, 4.0], [47.2, 6.1], [70.2, 9.8], [77.4, 35.2], [36.3, 59.4], [20.6, 41.7],
    [18.1, 19.0], [21.4, 13.0], [22.0, 8.3], [25.4, 9.1], [27.1, 7.4], [40.3, 8.0],
    [57.0, 12.3], [76.6, 19.5], [52.3, 45.7], [19.5, 51.1], [11.2, 29.7], [7.6, 15.8],
    [14.6, 9.7], [16.2, 10.1], [24.7, 8.6]])


def hudson_bay():
    """Реальная выборка Hudson Bay; целевая модель — Лотка–Вольтерра с параметрами ЛР № 1–3."""
    a, b, g, d = 0.481, 0.0248, 0.926, 0.0275
    Z, W = _truth([{(1, 0): a, (1, 1): -b}, {(1, 1): d, (0, 1): -g}], 2)
    return Sample("hudson_bay", "lotka", np.arange(21, dtype=float), HUDSON_BAY.copy(),
                  ("x", "y"), 2, Z, W, real=True)
