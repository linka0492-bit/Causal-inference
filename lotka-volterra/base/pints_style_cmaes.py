"""
Обёртка модели Volterra-Lotka в интерфейсе pints.ForwardModel и
самостоятельная реализация CMA-ES (алгоритм Hansen, "The CMA Evolution
Strategy: A Tutorial").

Почему не просто `import pints`: в этом контейнере отключена сеть
(network_configuration.Enabled = false), поэтому `pip install pints`
и `pip install cma` недоступны -- проверено, pip возвращает
"No matching distribution found". Чтобы не блокироваться на этом,
класс LotkaVolterraForwardModel реализует ровно два метода, которые
требует pints.ForwardModel (n_parameters, simulate) -- если запустить
этот же файл в среде с интернетом, достаточно сделать класс наследником
pints.ForwardModel и заменить SimpleCMAES на pints.OptimisationController
с pints.CMAES -- остальной код не изменится.
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from lotka_volterra_tree import build_lotka_volterra, ExpressionTree


class LotkaVolterraForwardModel:
    """
    Совместим по интерфейсу с pints.ForwardModel:
      - n_parameters() -> int
      - simulate(parameters, times) -> np.ndarray формы (len(times), n_outputs)
    Модель компилируется из дерева/DAG, построенного в lotka_volterra_tree.py,
    а не задаётся заново вручную.
    """

    def __init__(self):
        built = build_lotka_volterra()
        tree: ExpressionTree = built["tree"]

        dxdt_expr = tree.to_sympy(built["dxdt_root"])
        dydt_expr = tree.to_sympy(built["dydt_root"])

        x_s, y_s = sp.symbols("x y")
        alpha_s, beta_s, gamma_s, delta_s = sp.symbols("alpha beta gamma delta")

        self._dxdt = sp.lambdify((x_s, y_s, alpha_s, beta_s, gamma_s, delta_s), dxdt_expr, "numpy")
        self._dydt = sp.lambdify((x_s, y_s, alpha_s, beta_s, gamma_s, delta_s), dydt_expr, "numpy")

    def n_parameters(self) -> int:
        # alpha, beta, gamma, delta, x0, y0
        return 6

    def n_outputs(self) -> int:
        return 2

    def simulate(self, parameters, times) -> np.ndarray:
        alpha, beta, gamma, delta, x0, y0 = parameters

        def rhs(t, state):
            x_val, y_val = state
            return [
                self._dxdt(x_val, y_val, alpha, beta, gamma, delta),
                self._dydt(x_val, y_val, alpha, beta, gamma, delta),
            ]

        sol = solve_ivp(
            rhs, t_span=(times[0], times[-1]), y0=[x0, y0],
            t_eval=times, method="RK45", rtol=1e-8, atol=1e-8,
        )
        if not sol.success or sol.y.shape[1] != len(times):
            # штраф за "развалившуюся" траекторию -- нужно для устойчивости CMA-ES
            return np.full((len(times), 2), 1e6)
        return sol.y.T  # (len(times), 2)


class ProblemErrorMeasure:
    """
    Аналог pints.SumOfSquaresError поверх ForwardModel: считает сумму
    квадратов отклонений модели от наблюдаемых данных по обоим выходам.
    """

    def __init__(self, model: LotkaVolterraForwardModel, times, data):
        self._model = model
        self._times = times
        self._data = data  # shape (len(times), 2)

    def __call__(self, parameters) -> float:
        sim = self._model.simulate(parameters, self._times)
        return float(np.sum((sim - self._data) ** 2))


class SimpleCMAES:
    """
    Минимальная реализация (mu/mu_w, lambda)-CMA-ES по Hansen (2016),
    "The CMA Evolution Strategy: A Tutorial", алгоритм 1 (упрощённая версия
    без активного обновления C через rank-1/rank-mu раздельно -- здесь
    объединённое rank-mu обновление для компактности).
    """

    def __init__(self, x0, sigma0, bounds=None, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n = len(x0)
        self.mean = np.array(x0, dtype=float)
        self.sigma = float(sigma0)
        self.bounds = bounds  # список (low, high) для каждой координаты, либо None

        # стандартные параметры популяции (формулы Hansen'a)
        self.lam = 4 + int(3 * np.log(self.n))
        self.mu = self.lam // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights / np.sum(weights)
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs

        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.B = np.eye(self.n)
        self.D = np.ones(self.n)
        self.C = np.eye(self.n)
        self.chiN = self.n ** 0.5 * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

    def _clip(self, x):
        if self.bounds is None:
            return x
        return np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(x, self.bounds)])

    def run(self, objective, n_generations=150, verbose=True):
        best_x, best_f = self.mean.copy(), objective(self.mean)

        for gen in range(n_generations):
            # генерация популяции
            Z = self.rng.standard_normal((self.lam, self.n))
            Y = Z @ np.diag(self.D) @ self.B.T
            X = np.array([self._clip(self.mean + self.sigma * y) for y in Y])

            fitness = np.array([objective(x) for x in X])
            order = np.argsort(fitness)
            X, Y, fitness = X[order], Y[order], fitness[order]

            if fitness[0] < best_f:
                best_f, best_x = fitness[0], X[0].copy()

            # обновление среднего
            X_mu, Y_mu = X[:self.mu], Y[:self.mu]
            new_mean = np.sum(self.weights[:, None] * X_mu, axis=0)
            y_w = (new_mean - self.mean) / self.sigma
            self.mean = new_mean

            # эволюционные пути
            C_inv_sqrt = self.B @ np.diag(1.0 / self.D) @ self.B.T
            self.ps = (1 - self.cs) * self.ps + \
                np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (C_inv_sqrt @ y_w)
            hsig = np.linalg.norm(self.ps) / np.sqrt(
                1 - (1 - self.cs) ** (2 * (gen + 1))) / self.chiN < 1.4 + 2 / (self.n + 1)
            self.pc = (1 - self.cc) * self.pc + \
                hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

            # обновление ковариационной матрицы (rank-1 + rank-mu)
            artmp = Y_mu
            self.C = (
                (1 - self.c1 - self.cmu) * self.C
                + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                + self.cmu * (artmp.T * self.weights) @ artmp
            )

            # адаптация шага
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

            # обновление собственного разложения C (стабилизация)
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.clip(eigvals, 1e-20, None)
            self.D = np.sqrt(eigvals)
            self.B = eigvecs

            if verbose and gen % 20 == 0:
                print(f"  поколение {gen:4d}: best_f={best_f:.3f}  sigma={self.sigma:.4f}")

        return best_x, best_f
