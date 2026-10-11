"""
Шаг 3 эксперимента: настройка параметров дерева Volterra-Lotka
(построенного в lotka_volterra_tree.py) методом Левенберга-Марквардта
на реальной выборке — данные Hudson Bay Company, 1900-1920
(численность зайца-беляка и рыси, в тысячах особей/шкур).

Источник данных: Hudson Bay Company pelt records, воспроизведены во
множестве источников, в т.ч. Odum, "Fundamentals of Ecology", p.191;
идентичная таблица используется как toy-датасет в pints.toy.LotkaVolterraModel.

Метод: scipy.optimize.least_squares(method="lm") — это и есть классический
Левенберг-Марквардт для задачи наименьших квадратов (эквивалент
scipy.optimize.curve_fit, который под капотом вызывает тот же MINPACK LMDIF).
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from lotka_volterra_tree import build_lotka_volterra, ExpressionTree


# ---------------------------------------------------------------------------
# 1. Реальная выборка (Hudson Bay Company, 1900-1920)
# ---------------------------------------------------------------------------

YEARS = np.arange(1900, 1921)
HARE = np.array([
    30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
    27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7,
])
LYNX = np.array([
    4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1,
    7.4, 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6,
])

T = YEARS - YEARS[0]  # время в годах от начала наблюдений: 0..20


# ---------------------------------------------------------------------------
# 2. Компиляция дерева в численную правую часть, параметризованную вектором theta
# ---------------------------------------------------------------------------

def make_rhs_from_tree():
    """
    Берёт дерево/DAG из lotka_volterra_tree.build_lotka_volterra(), компилирует
    его в SymPy-выражения и возвращает функцию rhs(t, state, theta), где
    theta = [alpha, beta, gamma, delta] — как раз то, что будет подгонять LM.
    """
    built = build_lotka_volterra()
    tree: ExpressionTree = built["tree"]

    dxdt_expr = tree.to_sympy(built["dxdt_root"])
    dydt_expr = tree.to_sympy(built["dydt_root"])

    x_s, y_s = sp.symbols("x y")
    alpha_s, beta_s, gamma_s, delta_s = sp.symbols("alpha beta gamma delta")

    # важно: НЕ подставляем числа в параметры — оставляем их как символы,
    # чтобы lambdify дал функцию от (x, y, alpha, beta, gamma, delta).
    dxdt_num = sp.lambdify((x_s, y_s, alpha_s, beta_s, gamma_s, delta_s), dxdt_expr, "numpy")
    dydt_num = sp.lambdify((x_s, y_s, alpha_s, beta_s, gamma_s, delta_s), dydt_expr, "numpy")

    def rhs(t, state, theta):
        x_val, y_val = state
        alpha, beta, gamma, delta = theta
        return [
            dxdt_num(x_val, y_val, alpha, beta, gamma, delta),
            dydt_num(x_val, y_val, alpha, beta, gamma, delta),
        ]

    return rhs, built["params"]


# ---------------------------------------------------------------------------
# 3. Функция невязки (residuals) для least_squares
# ---------------------------------------------------------------------------

def simulate(theta, t_eval, y0):
    rhs, _ = SIMULATE_CACHE["rhs"], None
    sol = solve_ivp(
        SIMULATE_CACHE["rhs"], t_span=(t_eval[0], t_eval[-1]), y0=y0,
        t_eval=t_eval, args=(theta,), method="RK45", rtol=1e-8, atol=1e-8,
    )
    return sol.y  # shape (2, len(t_eval))


def residuals(params_vec, t_eval, data_hare, data_lynx):
    """
    params_vec = [alpha, beta, gamma, delta, x0, y0] — подгоняем не только
    параметры уравнения, но и начальные условия (x0, y0), т.к. реальная
    выборка не задаёт точный x(0), y(0) заранее.
    """
    alpha, beta, gamma, delta, x0, y0 = params_vec
    theta = (alpha, beta, gamma, delta)
    sim = simulate(theta, t_eval, [x0, y0])
    res_x = sim[0] - data_hare
    res_y = sim[1] - data_lynx
    return np.concatenate([res_x, res_y])


def main():
    rhs, param_nodes = make_rhs_from_tree()
    SIMULATE_CACHE["rhs"] = rhs

    # стартовая точка: значения параметров, зашитые в дерево на шаге 2,
    # плюс начальные условия — берём просто первые точки выборки
    theta0 = [param_nodes[name].value for name in ("alpha", "beta", "gamma", "delta")]
    x0_0, y0_0 = HARE[0], LYNX[0]
    p0 = np.array(theta0 + [x0_0, y0_0])

    print("Стартовая точка (из дерева, шаг 2):")
    print(f"  alpha={p0[0]:.4f} beta={p0[1]:.4f} gamma={p0[2]:.4f} delta={p0[3]:.4f} "
          f"x0={p0[4]:.2f} y0={p0[5]:.2f}")

    result = least_squares(
        residuals, p0, method="lm", max_nfev=20000,
        args=(T.astype(float), HARE, LYNX),
    )

    alpha, beta, gamma, delta, x0, y0 = result.x
    print("\nПодобранные параметры (Левенберг-Марквардт, scipy MINPACK LMDIF):")
    print(f"  alpha={alpha:.4f}  beta={beta:.4f}  gamma={gamma:.4f}  delta={delta:.4f}")
    print(f"  x0={x0:.2f}  y0={y0:.2f}")
    print(f"\nУспех оптимизации: {result.success}, сообщение: {result.message}")
    print(f"Число вызовов функции невязки: {result.nfev}")

    rss = np.sum(result.fun ** 2)
    print(f"\nОстаточная сумма квадратов (RSS): {rss:.2f}")

    # сравнение модели с данными по годам
    sim_final = simulate((alpha, beta, gamma, delta), T.astype(float), [x0, y0])
    print("\n  Год  |  Hare(данные)  Hare(модель)  |  Lynx(данные)  Lynx(модель)")
    for i, year in enumerate(YEARS):
        print(f"  {year} |  {HARE[i]:12.1f}  {sim_final[0, i]:12.2f}  |  "
              f"{LYNX[i]:12.1f}  {sim_final[1, i]:12.2f}")


SIMULATE_CACHE = {"rhs": None}


if __name__ == "__main__":
    main()
