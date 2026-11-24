"""Настройка параметров модели.

Два режима:
1) ИНТЕГРАЛЬНОЕ СОГЛАСОВАНИЕ (быстро, используется внутри ГА). Данные интерполируются (PCHIP),
   и для каждого интервала [t_i, t_{i+1}] требуется  x_{i+1} − x_i ≈ ∫ f(x̂(t), ŷ(t); θ) dt.
   Не нужно решать ОДУ, нет дифференцирования зашумлённых данных, дерево подгоняется отдельно
   для каждого уравнения (константы разных деревьев независимы).
2) ПОДГОНКА ТРАЕКТОРИИ (точно, для итоговых моделей и графиков): решаем ОДУ и минимизируем
   RSS = Σ (x_obs − x_sim)² + Σ (y_obs − y_sim)², параметры θ и начальные условия x0, y0.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from .expr import Model, Node, evaluate, consts, InvalidTree, size

_GL_X, _GL_W = np.polynomial.legendre.leggauss(6)
BAD = 1e3  # значение невязки для некорректных параметров


class IntegralProblem:
    """Предвычисленная задача интегрального согласования для набора данных."""

    def __init__(self, ds):
        t, x, y = ds.t, ds.x, ds.y
        px, py = PchipInterpolator(t, x), PchipInterpolator(t, y)
        tm = 0.5 * (t[1:] + t[:-1])[:, None]
        h = 0.5 * (t[1:] - t[:-1])[:, None]
        tq = tm + h * _GL_X[None, :]
        self.W = h * _GL_W[None, :]
        self.X, self.Y = px(tq), py(tq)
        self.env = {"x": self.X.ravel(), "y": self.Y.ravel()}
        self.shape = self.X.shape
        self.tgt = [x[1:] - x[:-1], y[1:] - y[:-1]]
        self.scale = [max(np.sqrt(np.mean(g ** 2)), 1e-9) for g in self.tgt]
        self.sst = [max(np.sum((g - g.mean()) ** 2), 1e-12) for g in self.tgt]
        self.n = ds.n

    def pred(self, tree):
        f = evaluate(tree, self.env, check=True).reshape(self.shape)
        return (self.W * f).sum(1)

    def residual(self, tree, k):
        return (self.tgt[k] - self.pred(tree)) / self.scale[k]

    def rel_err(self, tree, k):
        """SSE / SST по приращениям (аналог 1 − R²)."""
        try:
            r = self.residual(tree, k) * self.scale[k]
        except InvalidTree:
            return np.inf
        return float(np.sum(r ** 2) / self.sst[k])


def fit_tree(tree: Node, prob: IntegralProblem, k: int, restarts=0, rng=None, max_nfev=60):
    """Подгонка констант одного дерева (на месте). Возвращает rel_err (inf, если дерево некорректно)."""
    cs = consts(tree)
    if not cs:
        return prob.rel_err(tree, k)
    rng = rng or np.random.default_rng(0)

    def set_theta(th):
        for c, v in zip(cs, th):
            c.val = float(v)

    def fun(th):
        set_theta(th)
        try:
            return prob.residual(tree, k)
        except InvalidTree:
            return np.full(prob.n - 1, BAD)

    starts = [np.array([c.val for c in cs])]
    for _ in range(restarts):
        starts.append(np.array([np.sign(rng.normal()) * 10 ** rng.uniform(-2, 0.5) for _ in cs]))
    best, best_cost = starts[0], np.inf
    for th0 in starts:
        try:
            r0 = fun(th0)
            if np.max(np.abs(r0)) >= BAD:
                if len(starts) == 1:
                    continue
            sol = least_squares(fun, th0, method="trf", bounds=(-1e3, 1e3), max_nfev=max_nfev,
                                x_scale="jac")
            if sol.cost < best_cost and np.max(np.abs(sol.fun)) < BAD:
                best, best_cost = sol.x, sol.cost
        except Exception:
            continue
    if not np.isfinite(best_cost):
        set_theta(starts[0])
        return np.inf
    set_theta(best)
    return prob.rel_err(tree, k)


def fit_model(model: Model, prob: IntegralProblem, restarts=0, rng=None, max_nfev=60):
    """Подгонка обоих деревьев. Возвращает (rel_x, rel_y)."""
    return tuple(fit_tree(t, prob, k, restarts, rng, max_nfev) for k, t in enumerate(model.trees))


# ----------------------------------------------------------------------------
# Траектория ОДУ
# ----------------------------------------------------------------------------
def simulate(model: Model, x0, y0, t_eval, t0=None, rtol=1e-7, atol=1e-8, xmax=1e5):
    """Решение ОДУ. Возвращает массив (len(t_eval), 2) или None, если модель «взорвалась»/некорректна."""
    t_eval = np.asarray(t_eval, float)
    t0 = t_eval[0] if t0 is None else t0

    def rhs(t, s):
        try:
            a, b = model.rhs(np.array([s[0]]), np.array([s[1]]), check=True)
            return [float(a[0]), float(b[0])]
        except InvalidTree:
            return [np.nan, np.nan]

    def blow(t, s):
        return xmax - max(abs(s[0]), abs(s[1]))
    blow.terminal = True
    try:
        sol = solve_ivp(rhs, (t0, t_eval[-1]), [x0, y0], t_eval=t_eval, method="LSODA",
                        rtol=rtol, atol=atol, events=blow, max_step=0.25)
    except Exception:
        return None
    if (not sol.success) or sol.y.shape[1] != len(t_eval) or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y.T


def rss(model: Model, ds, x0=None, y0=None):
    x0 = ds.x[0] if x0 is None else x0
    y0 = ds.y[0] if y0 is None else y0
    sim = simulate(model, x0, y0, ds.t)
    if sim is None:
        return np.inf
    return float(np.sum((ds.x - sim[:, 0]) ** 2) + np.sum((ds.y - sim[:, 1]) ** 2))


def fit_trajectory(model: Model, ds, fit_init=True, max_nfev=60, restarts=0, rng=None):
    """Подгонка θ (и x0,y0) по траектории ОДУ (метод trf). Изменяет model на месте.
    Возвращает dict(rss, x0, y0, success)."""
    cs = model.const_nodes()
    th0 = model.get_params()
    p0 = np.concatenate([th0, [ds.x[0], ds.y[0]]]) if fit_init else th0

    def unpack(p):
        th = p[:len(cs)]
        x0, y0 = (p[-2], p[-1]) if fit_init else (ds.x[0], ds.y[0])
        return th, x0, y0

    def fun(p):
        th, x0, y0 = unpack(p)
        model.set_params(th)
        sim = simulate(model, x0, y0, ds.t, rtol=1e-6, atol=1e-7)
        if sim is None:
            return np.full(2 * ds.n, 1e3)
        return np.concatenate([ds.x - sim[:, 0], ds.y - sim[:, 1]])

    rng = rng or np.random.default_rng(0)
    starts = [p0]
    for _ in range(restarts):
        q = p0.copy(); q[:len(cs)] *= np.exp(rng.normal(0, 0.3, len(cs))); starts.append(q)
    best_p, best_c = p0, np.inf
    for s in starts:
        try:
            lb = np.concatenate([np.full(len(cs), -1e3), [0.1, 0.1]]) if fit_init else np.full(len(cs), -1e3)
            ub = np.concatenate([np.full(len(cs), 1e3), [500, 500]]) if fit_init else np.full(len(cs), 1e3)
            sol = least_squares(fun, np.clip(s, lb + 1e-9, ub - 1e-9), method="trf", bounds=(lb, ub),
                                max_nfev=max_nfev, x_scale="jac")
            if sol.cost < best_c:
                best_p, best_c = sol.x, sol.cost
        except Exception:
            pass
    th, x0, y0 = unpack(best_p)
    model.set_params(th)
    val = rss(model, ds, x0, y0)
    model.meta["x0"], model.meta["y0"] = float(x0), float(y0)
    return {"rss": val, "x0": float(x0), "y0": float(y0), "success": bool(np.isfinite(val))}


def minibatch_params(model: Model, ds, n_batches=100, k=12, seed=0, max_nfev=30):
    """Распределение параметров по мини-батчам (процедура A из ЛР № 2) на интегральном согласовании.
    Возвращает массив (n_batches, n_params)."""
    from .data import Dataset
    rng = np.random.default_rng(seed)
    th0 = model.get_params()
    out = []
    m = model.copy()
    for _ in range(n_batches):
        idx = np.sort(rng.choice(ds.n, min(k, ds.n), replace=False))
        sub = Dataset(ds.t[idx], ds.x[idx], ds.y[idx])
        prob = IntegralProblem(sub)
        m.set_params(th0)
        fit_model(m, prob, max_nfev=max_nfev)
        out.append(m.get_params())
    return np.array(out)
