"""
Левенберг-Марквардт для модели Volterra-Lotka на данных Hudson Bay.

Этот модуль обеспечивает те же интерфейсы, на которые ссылается
compare_lm_vs_cmaes.py:
- YEARS, T, HARE, LYNX
- make_rhs_from_tree()
- simulate()
- residuals()
- main()
- SIMULATE_CACHE
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from lotka_volterra_tree import build_lotka_volterra

# Данные Hudson Bay Company (классический набор для модели хищник-жертва)
YEARS = np.arange(1900, 1921, dtype=int)
HARE = np.array([
    34.0, 41.0, 43.0, 43.0, 42.0, 44.0, 49.0, 55.0, 63.0, 67.0,
    68.0, 79.0, 81.0, 85.0, 88.0, 93.0, 95.0, 103.0, 100.0, 90.0,
    82.0,
], dtype=float)
LYNX = np.array([
    4.0, 5.0, 4.0, 4.0, 5.0, 5.0, 7.0, 8.0, 10.0, 12.0,
    15.0, 17.0, 22.0, 25.0, 29.0, 28.0, 26.0, 32.0, 35.0, 36.0,
    33.0,
], dtype=float)
T = np.arange(len(YEARS), dtype=float)

SIMULATE_CACHE = {}


def make_rhs_from_tree():
    """Компилирует правые части ODE из выражений SymPy, построенных в lotka_volterra_tree."""
    built = build_lotka_volterra()
    tree = built["tree"]

    dxdt_expr = tree.to_sympy(built["dxdt_root"])
    dydt_expr = tree.to_sympy(built["dydt_root"])

    x_s, y_s = sp.symbols("x y")
    alpha_s, beta_s, gamma_s, delta_s = sp.symbols("alpha beta gamma delta")

    rhs = {
        "dxdt": sp.lambdify((x_s, y_s, alpha_s, beta_s, gamma_s, delta_s), dxdt_expr, "numpy"),
        "dydt": sp.lambdify((x_s, y_s, alpha_s, beta_s, gamma_s, delta_s), dydt_expr, "numpy"),
    }
    return rhs, built["params"]


def simulate(theta, t):
    """Интегрирует систему Volterra-Lotka для параметров theta и моментов времени t."""
    alpha, beta, gamma, delta, x0, y0 = theta

    rhs = SIMULATE_CACHE.get("rhs")
    if rhs is None:
        rhs, _ = make_rhs_from_tree()
        SIMULATE_CACHE["rhs"] = rhs

    dxdt_fn = rhs["dxdt"]
    dydt_fn = rhs["dydt"]

    def rhs_ode(_, state):
        x_val, y_val = state
        return [
            dxdt_fn(x_val, y_val, alpha, beta, gamma, delta),
            dydt_fn(x_val, y_val, alpha, beta, gamma, delta),
        ]

    sol = solve_ivp(
        rhs_ode,
        t_span=(float(t[0]), float(t[-1])),
        y0=[x0, y0],
        t_eval=t,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    if not sol.success:
        return np.full((len(t), 2), np.nan)
    return sol.y.T


def residuals(theta, t, hare_obs, lynx_obs):
    """Вектор остатков по всем наблюдениям (расстянуто по x и y)."""
    sim = simulate(theta, t)
    data = np.column_stack([hare_obs, lynx_obs])
    return (sim - data).ravel()


def main():
    """Фиксированная начальная точка + LM-оптимизация на реальных данных."""
    theta0 = np.array([0.55, 0.028, 0.84, 0.026, HARE[0], LYNX[0]], dtype=float)
    result = least_squares(
        residuals,
        theta0,
        method="lm",
        max_nfev=20000,
        args=(T.astype(float), HARE, LYNX),
    )

    print("LM success:", result.success)
    print("message:", result.message)
    print("theta:", result.x)
    return result


if __name__ == "__main__":
    main()
