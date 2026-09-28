"""
Шаг 3, продолжение: та же задача (подгонка параметров Volterra-Lotka под
данные Hudson Bay), но методом CMA-ES вместо Левенберга-Марквардта,
через обёртку в стиле pints.ForwardModel (см. pints_style_cmaes.py и
пояснение там же про отсутствие сети в контейнере).

В конце -- визуализация: данные vs LM vs CMA-ES на одном графике.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_lotka_volterra_lm import YEARS, T, HARE, LYNX, main as run_lm_and_get_result
from pints_style_cmaes import LotkaVolterraForwardModel, ProblemErrorMeasure, SimpleCMAES


def run_lm():
    """Повторяем шаг с Левенбергом-Марквардтом, но возвращаем параметры, а не только печать."""
    from scipy.optimize import least_squares
    from fit_lotka_volterra_lm import make_rhs_from_tree, simulate, residuals, SIMULATE_CACHE

    rhs, param_nodes = make_rhs_from_tree()
    SIMULATE_CACHE["rhs"] = rhs

    theta0 = [param_nodes[name].value for name in ("alpha", "beta", "gamma", "delta")]
    p0 = np.array(theta0 + [HARE[0], LYNX[0]])

    result = least_squares(residuals, p0, method="lm", max_nfev=20000, args=(T.astype(float), HARE, LYNX))
    return result.x  # alpha, beta, gamma, delta, x0, y0


def run_cmaes():
    model = LotkaVolterraForwardModel()
    data = np.column_stack([HARE, LYNX])
    error = ProblemErrorMeasure(model, T.astype(float), data)

    # стартовая точка -- те же значения, что зашиты в дерево на шаге 2
    x0_real = np.array([0.55, 0.028, 0.84, 0.026, HARE[0], LYNX[0]])

    # ВАЖНО: масштабы параметров различаются на порядки (alpha ~ 0.5,
    # beta ~ 0.03, x0 ~ 30) -- при общем изотропном шаге sigma это ломает
    # CMA-ES (шаг либо слишком велик для beta/delta, либо слишком мал для
    # начальных условий). Поэтому оптимизируем в нормированных координатах
    # z = theta / scales, где scales -- сама стартовая точка: тогда z0 = 1
    # по всем координатам и шаг sigma0 одинаково осмыслен для всех.
    scales = x0_real.copy()

    def error_normalized(z):
        theta_real = z * scales
        # мягкий барьер вместо жёсткого clip -- не даёт градиенту "застревать"
        # на границе допустимой области
        if np.any(theta_real[:4] <= 0) or np.any(theta_real[4:] <= 0):
            return 1e8
        return error(theta_real)

    z0 = np.ones(6)
    bounds_z = [(1e-3, 5.0)] * 4 + [(0.1, 5.0)] * 2

    print("Запуск CMA-ES (собственная реализация, алгоритм Hansen) в нормированных координатах...")
    optimiser = SimpleCMAES(x0=z0, sigma0=0.3, bounds=bounds_z, seed=42)
    best_z, best_f = optimiser.run(error_normalized, n_generations=200, verbose=True)

    best_x = best_z * scales
    print(f"\nCMA-ES завершён. Итоговая ошибка (RSS): {best_f:.2f}")
    return best_x


def main():
    print("=" * 70)
    print("1) Левенберг-Марквардт (повтор)")
    print("=" * 70)
    lm_params = run_lm()
    alpha_lm, beta_lm, gamma_lm, delta_lm, x0_lm, y0_lm = lm_params
    print(f"  alpha={alpha_lm:.4f} beta={beta_lm:.4f} gamma={gamma_lm:.4f} delta={delta_lm:.4f} "
          f"x0={x0_lm:.2f} y0={y0_lm:.2f}")

    print("\n" + "=" * 70)
    print("2) CMA-ES")
    print("=" * 70)
    cmaes_params = run_cmaes()
    alpha_cma, beta_cma, gamma_cma, delta_cma, x0_cma, y0_cma = cmaes_params
    print(f"  alpha={alpha_cma:.4f} beta={beta_cma:.4f} gamma={gamma_cma:.4f} delta={delta_cma:.4f} "
          f"x0={x0_cma:.2f} y0={y0_cma:.2f}")

    # -----------------------------------------------------------------
    # 3) Визуализация: данные vs LM vs CMA-ES
    # -----------------------------------------------------------------
    model = LotkaVolterraForwardModel()
    t_dense = np.linspace(T[0], T[-1], 300)

    sim_lm = model.simulate(lm_params, t_dense)
    sim_cma = model.simulate(cmaes_params, t_dense)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))  # разные смыслы оси X у графиков -- sharex не нужен

    ax = axes[0]
    ax.scatter(YEARS, HARE, color="tab:blue", label="Заяц -- данные (Hudson Bay)", zorder=5)
    ax.plot(YEARS[0] + t_dense, sim_lm[:, 0], color="tab:blue", linestyle="--", label="Заяц -- LM")
    ax.plot(YEARS[0] + t_dense, sim_cma[:, 0], color="tab:blue", linestyle="-", alpha=0.7, label="Заяц -- CMA-ES")
    ax.scatter(YEARS, LYNX, color="tab:orange", label="Рысь -- данные (Hudson Bay)", zorder=5)
    ax.plot(YEARS[0] + t_dense, sim_lm[:, 1], color="tab:orange", linestyle="--", label="Рысь -- LM")
    ax.plot(YEARS[0] + t_dense, sim_cma[:, 1], color="tab:orange", linestyle="-", alpha=0.7, label="Рысь -- CMA-ES")
    ax.set_ylabel("Численность (тыс.)")
    ax.set_title("Volterra-Lotka: данные Hudson Bay (1900-1920) vs подгонка LM и CMA-ES")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # фазовый портрет (x vs y) -- удобно, чтобы увидеть форму предельного цикла
    ax2 = axes[1]
    ax2.plot(HARE, LYNX, "o-", color="gray", alpha=0.6, label="данные")
    ax2.plot(sim_lm[:, 0], sim_lm[:, 1], "--", color="tab:blue", label="LM")
    ax2.plot(sim_cma[:, 0], sim_cma[:, 1], "-", color="tab:red", alpha=0.8, label="CMA-ES")
    ax2.set_xlabel("Заяц (тыс.)")
    ax2.set_ylabel("Рысь (тыс.)")
    ax2.set_title("Фазовый портрет")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lotka_volterra_fit_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nГрафик сохранён: {out_path}")

    # -----------------------------------------------------------------
    # 4) Итоговое сравнение методов
    # -----------------------------------------------------------------
    rss_lm = np.sum((model.simulate(lm_params, T.astype(float)) - np.column_stack([HARE, LYNX])) ** 2)
    rss_cma = np.sum((model.simulate(cmaes_params, T.astype(float)) - np.column_stack([HARE, LYNX])) ** 2)

    print("\n" + "=" * 70)
    print("Итоговое сравнение (RSS на реальных точках наблюдений)")
    print("=" * 70)
    print(f"  Левенберг-Марквардт: RSS = {rss_lm:.2f}")
    print(f"  CMA-ES:              RSS = {rss_cma:.2f}")


if __name__ == "__main__":
    main()
