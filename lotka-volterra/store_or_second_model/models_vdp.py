"""
Вторая модель: осциллятор Ван-дер-Поля на СИНТЕТИЧЕСКИХ данных (истинные параметры известны).

    dx/dt = v
    dv/dt = mu * (1 - x^2) * v - omega^2 * x

Правая часть строится тем же формализмом, что и для Лотки-Вольтерры: DAG из операций
{+,-,*,/} (три цвета вершин), компилируется compile_tree. Параметры подгонки:
theta = (mu, omega, x0, v0). Истина: mu=1.5, omega=1.0, x0=2.0, v0=0.0.
"""

import numpy as np
from scipy.integrate import solve_ivp

import lv_common  # noqa: F401  (заглушка sympy)
from lv_common import compile_tree
from lotka_volterra_tree import ExpressionTree

VDP_NAMES = ["mu", "omega", "x0", "v0"]
TRUTH = np.array([1.5, 1.0, 2.0, 0.0])
LOWER = np.array([0.01, 0.1, -10.0, -10.0])
UPPER = np.array([10.0, 5.0, 10.0, 10.0])
P_START = np.array([1.0, 0.9, 1.5, 0.3])  # намеренно не истина
T_GRID = np.linspace(0.0, 20.0, 60)
NOISE_SD = 0.15


def build_vdp_tree():
    tree = ExpressionTree()
    x = tree.observed("x")
    v = tree.observed("y")  # вторая наблюдаемая (скорость); имя входа в compile_tree -- "y"
    mu = tree.parameter("mu", 1.5)
    w2 = tree.parameter("w2", 1.0)
    one = tree.parameter("one", 1.0)  # константа 1 как параметр-вход (словарь операций без констант)
    xx = tree.op("*", x, x)
    om = tree.op("-", one, xx)
    t1 = tree.op("*", mu, om)
    t2 = tree.op("*", t1, v)
    t3 = tree.op("*", w2, x)
    dv = tree.op("-", t2, t3)
    return tree, (v, dv)  # dx/dt = v (корень -- сама наблюдаемая), dv/dt = dv


class VdPModel:
    def __init__(self, rtol=1e-6, atol=1e-6):
        self.tree, roots = build_vdp_tree()
        self.f, self.src = compile_tree(self.tree, roots, input_names=["x", "y", "mu", "w2", "one"])
        self.rtol, self.atol = rtol, atol

    def simulate(self, p, t_eval):
        t_eval = np.asarray(t_eval, dtype=float)
        mu, omega, x0, v0 = p
        w2 = omega ** 2

        def rhs(t, s):
            return self.f(s[0], s[1], mu, w2, 1.0)

        sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [x0, v0], t_eval=t_eval,
                        method="RK45", rtol=self.rtol, atol=self.atol)
        if (not sol.success) or sol.y.shape[1] != len(t_eval):
            return np.full((len(t_eval), 2), 1e6)
        return sol.y.T


def synthetic_data(model, seed=123):
    rng = np.random.default_rng(seed)
    clean = model.simulate(TRUTH, T_GRID)
    return clean + rng.normal(0, NOISE_SD, size=clean.shape)
