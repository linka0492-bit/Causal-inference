"""
Общие утилиты для интервенций и мини-батч экспериментов.

Что здесь есть:
  * заглушка для sympy (в контейнере нет сети, pip его поставить не может;
    а lotka_volterra_tree.py делает `import sympy`, хотя использует его только
    в to_sympy()). Если sympy есть -- он используется как обычно;
  * compile_tree: компиляция DAG-дерева суперпозиции в обычную python-функцию
    (кодогенерация по узлам в топологическом порядке) -- без sympy и быстрее lambdify;
  * build_lv_with_love: то же дерево Volterra-Lotka + ИНТЕРВЕНЦИЯ «взаимная любовь»;
  * LVModel: интегрирование системы с возможностью включить интервенцию в момент t_on;
  * fit: подгонка параметров (alpha, beta, gamma, delta, x0, y0) по подвыборке точек.
"""

import sys
import types

try:  # noqa: SIM105
    import sympy  # noqa: F401
except ImportError:
    sys.modules["sympy"] = types.ModuleType("sympy")  # to_sympy() здесь не вызывается

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from lotka_volterra_tree import build_lotka_volterra, ExpressionTree, NodeColor
from fit_lotka_volterra_lm import YEARS, T, HARE, LYNX  # те же данные Hudson Bay

DATA = np.column_stack([HARE, LYNX])  # (21, 2)
T_OBS = T.astype(float)

INPUT_NAMES = ["x", "y", "alpha", "beta", "gamma", "delta", "lam", "u"]
PARAM_NAMES = ["alpha", "beta", "gamma", "delta", "x0", "y0"]


# ---------------------------------------------------------------------------
# DAG -> python-функция
# ---------------------------------------------------------------------------

def compile_tree(tree: ExpressionTree, roots, input_names=INPUT_NAMES):
    """Возвращает f(*inputs) -> tuple значений корней. Узлы считаются один раз (DAG)."""
    needed = set()

    def mark(n):
        if n.id in needed:
            return
        needed.add(n.id)
        for p in n.parents:
            mark(p)

    for r in roots:
        mark(r)

    lines = []
    for n in tree.nodes:  # id растут от листьев к корням -> уже топологический порядок
        if n.id not in needed:
            continue
        if n.color == NodeColor.LATENT:
            a = [f"v{p.id}" for p in n.parents]
            if n.op == "+":
                e = " + ".join(a)
            elif n.op == "-":
                e = a[0] + "".join(f" - {s}" for s in a[1:])
            elif n.op == "*":
                e = " * ".join(a)
            else:
                e = f"{a[0]} / {a[1]}"
            lines.append(f"    v{n.id} = {e}")
        else:
            assert n.name in input_names, f"вход {n.name} не объявлен"
            lines.append(f"    v{n.id} = {n.name}")
    src = (f"def _f({', '.join(input_names)}):\n" + "\n".join(lines) +
           f"\n    return ({', '.join(f'v{r.id}' for r in roots)},)\n")
    ns = {}
    exec(src, ns)  # noqa: S102 -- код генерируется нами из дерева
    return ns["_f"], src


# ---------------------------------------------------------------------------
# Дерево с интервенцией
# ---------------------------------------------------------------------------

def build_lv_with_love(lam_nominal=0.014):
    """
    Базовая система (из lotka_volterra_tree.py):
        dx/dt = alpha*x - beta*x*y
        dy/dt = delta*x*y - gamma*y

    Интервенция «взаимная любовь» do(u=1): к ОБОИМ уравнениям добавляется один и тот же
    член встречи видов, т.е. каждая встреча зайца с рысью теперь ещё и выгодна ОБОИМ:
        dx/dt = alpha*x - beta*x*y + u*lam*x*y
        dy/dt = delta*x*y - gamma*y + u*lam*x*y
    (эквивалентно beta -> beta - lam, delta -> delta + lam при u=1: хищник слабее
    «ест», но сильнее «размножается» от контакта).

    u -- новая НАБЛЮДАЕМАЯ (управляющая) вершина: u(t) = 0 до момента вмешательства и 1 после.
    lam -- новый параметр. Узел love = lam*u*x*y общий для двух корней (DAG), а узел x*y
    переиспользуется ещё раз.
    """
    built = build_lotka_volterra()
    tree = built["tree"]

    xy = next(n for n in tree.nodes
              if n.color == NodeColor.LATENT and n.op == "*"
              and {p.name for p in n.parents if p.color == NodeColor.OBSERVED} == {"x", "y"})

    n_before = len(tree.nodes)
    u = tree.observed("u")
    lam = tree.parameter("lam", value=lam_nominal)
    lam_u = tree.op("*", lam, u)
    love = tree.op("*", lam_u, xy)
    dxdt_love = tree.op("+", built["dxdt_root"], love)
    dydt_love = tree.op("+", built["dydt_root"], love)
    new_ids = {n.id for n in tree.nodes[n_before:]}

    return {
        "tree": tree, "xy": xy, "love": love, "new_ids": new_ids,
        "roots_base": (built["dxdt_root"], built["dydt_root"]),
        "roots_love": (dxdt_love, dydt_love),
        "params": built["params"],
    }


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

class LVModel:
    """Интегратор системы, скомпилированной из дерева (с интервенцией или без)."""

    def __init__(self, rtol=1e-7, atol=1e-7):
        self.built = build_lv_with_love()
        self.f, self.src = compile_tree(self.built["tree"], self.built["roots_love"])
        self.rtol, self.atol = rtol, atol

    def _segment(self, p, lam, u, t0, t1, state0, pts):
        alpha, beta, gamma, delta = p[:4]
        f = self.f

        def rhs(t, s):
            return f(s[0], s[1], alpha, beta, gamma, delta, lam, u)

        sol = solve_ivp(rhs, (t0, t1), state0, t_eval=pts, method="RK45",
                        rtol=self.rtol, atol=self.atol)
        if (not sol.success) or sol.y.shape[1] != len(pts):
            return None
        return sol.y.T

    def simulate(self, p, t_eval, lam=0.0, t_on=None):
        """
        p = [alpha, beta, gamma, delta, x0, y0]. Интервенция включается в момент t_on
        (u=0 при t<t_on, u=1 при t>=t_on). t_on=None или lam=0 -- базовая модель.
        Возвращает массив (len(t_eval), 2); при развале траектории -- массив 1e6.
        """
        t_eval = np.asarray(t_eval, dtype=float)
        s0 = [p[4], p[5]]
        t_start, t_end = t_eval[0], t_eval[-1]
        bad = np.full((len(t_eval), 2), 1e6)

        if t_on is None or lam == 0.0 or t_on > t_end:
            out = self._segment(p, lam, 0.0, t_start, t_end, s0, t_eval)
            return bad if out is None else out
        if t_on <= t_start:
            out = self._segment(p, lam, 1.0, t_start, t_end, s0, t_eval)
            return bad if out is None else out

        pts1 = np.append(t_eval[t_eval < t_on], t_on)
        seg1 = self._segment(p, lam, 0.0, t_start, t_on, s0, pts1)
        if seg1 is None:
            return bad
        pts2 = t_eval[t_eval >= t_on]
        if len(pts2) == 0:
            return seg1[:-1]
        seg2 = self._segment(p, lam, 1.0, t_on, t_end, seg1[-1], pts2)
        if seg2 is None:
            return bad
        return np.vstack([seg1[:-1], seg2])


# ---------------------------------------------------------------------------
# Подгонка по (под)выборке
# ---------------------------------------------------------------------------

P_START = np.array([0.55, 0.028, 0.84, 0.026, HARE[0], LYNX[0]])  # старт из дерева (шаг 2)
LOWER = np.array([1e-6, 1e-6, 1e-6, 1e-6, 0.1, 0.1])
UPPER = np.array([5.0, 1.0, 5.0, 1.0, 300.0, 300.0])


def fit(model: LVModel, p0=P_START, idx=None, method="lm", max_nfev=None):
    """
    Метод наименьших квадратов по точкам с индексами idx (None -- вся выборка).
    Траектория всегда интегрируется на ПОЛНОЙ сетке времён, но невязки берутся
    только на точках подвыборки -- так мини-батч это просто подмножество наблюдений.
    """
    idx = np.arange(len(T_OBS)) if idx is None else np.asarray(idx)

    def resid(p):
        sim = model.simulate(p, T_OBS)
        return (sim - DATA)[idx].ravel()

    if method == "lm":
        res = least_squares(resid, p0, method="lm", max_nfev=max_nfev or 20000)
    else:
        res = least_squares(resid, np.clip(p0, LOWER * 1.0001, UPPER * 0.9999),
                            method="trf", bounds=(LOWER, UPPER), max_nfev=max_nfev or 200)
    return res.x, float(np.sum(res.fun ** 2)), res
