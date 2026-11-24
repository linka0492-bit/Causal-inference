"""Библиотека мономов и линейная подгонка уравнений.

Модель в этой работе — линейная по коэффициентам: dx_i/dt = sum_j w_ij * theta_j(x),
где theta_j — мономы степени <= DEGREE от наблюдаемых переменных. Структура модели —
бинарная матрица Z[i, j] (есть ли моном j в уравнении i); это матрица смежности
двудольного графа «мономы -> уравнения».

Коэффициенты подбираются без производных: интегральное согласование (как в ЛР № 3),
x_i(t_{k+1}) - x_i(t_k) = интеграл от правой части по интервалу, интеграл считается
по PCHIP-интерполяции данных.
"""
import itertools
from functools import lru_cache

import numpy as np
from scipy.interpolate import PchipInterpolator

MAX_VARS = 3
DEGREE = 3


# ---------------------------------------------------------------- мономы
@lru_cache(maxsize=None)
def make_library(n_vars: int, degree: int = DEGREE):
    """Все мономы степени <= degree от n_vars переменных (кортежи показателей степеней)."""
    terms = []
    for d in range(degree + 1):
        for e in itertools.product(range(d + 1), repeat=n_vars):
            if sum(e) == d:
                terms.append(tuple(e))
    return tuple(terms)


def term_name(e, names=("x", "y", "z")):
    """(1,1,0) -> 'x*y', (2,1,0) -> 'x^2*y', (0,0,0) -> '1'."""
    parts = []
    for v, p in enumerate(e):
        if p == 1:
            parts.append(names[v])
        elif p > 1:
            parts.append(f"{names[v]}^{p}")
    return "*".join(parts) if parts else "1"


def pad_term(e, n=MAX_VARS):
    return tuple(e) + (0,) * (n - len(e))


def term_structure(e, i_eq, n_eq):
    """Структурные признаки монома для уравнения i_eq (не зависят от названий переменных).

    degree, число различных переменных, содержит ли «свою» переменную x_i,
    содержит ли внешнюю (ненаблюдаемую уравнением) переменную-драйвер, константа ли.
    """
    deg = sum(e)
    distinct = sum(1 for p in e if p > 0)
    own = 1.0 if (i_eq < len(e) and e[i_eq] > 0) else 0.0
    driver = 1.0 if any(p > 0 for v, p in enumerate(e) if v >= n_eq) else 0.0
    return np.array([deg / 3.0, distinct / 3.0, own, driver, float(deg == 0)])


# ---------------------------------------------------------------- интегральное согласование
def integral_design(t, X, terms, nsub=10):
    """Матрица A[k, j] = интеграл monomial_j(x(t)) по k-му интервалу и приращения B[k, i].

    nsub — чётное число подынтервалов формулы Симпсона внутри каждого интервала.
    """
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    m = len(t) - 1
    u = np.linspace(0.0, 1.0, nsub + 1)
    tt = t[:-1, None] + (t[1:] - t[:-1])[:, None] * u[None, :]
    V = [PchipInterpolator(t, X[:, v])(tt) for v in range(X.shape[1])]
    wq = np.ones(nsub + 1)
    wq[1:-1:2] = 4.0
    wq[2:-1:2] = 2.0
    wq /= 3.0 * nsub
    h = t[1:] - t[:-1]
    A = np.empty((m, len(terms)))
    for j, e in enumerate(terms):
        T = np.ones_like(tt)
        for v, p in enumerate(e):
            if p:
                T = T * V[v] ** p
        A[:, j] = h * (T @ wq)
    B = X[1:] - X[:-1]
    return A, B


# ---------------------------------------------------------------- одно уравнение
class Equation:
    """Нормированная задача наименьших квадратов  b ≈ A w  для одного уравнения.

    Столбцы A и вектор b нормируются на евклидову длину, поэтому |w| — это «вклад»
    монома в приращение (безразмерный), а RSS — доля необъяснённого (||b||^2 = 1).
    """

    def __init__(self, A, b, lam=1e-4):
        self.sA = np.linalg.norm(A, axis=0)
        self.sA[self.sA == 0] = 1.0
        nb = np.linalg.norm(b)
        self.sb = nb if nb > 0 else 1.0
        self.A = A / self.sA
        self.b = b / self.sb
        self.lam = lam
        self.m, self.K = self.A.shape

    # --- подгонка
    def fit(self, active, w0=None, prior=0.0):
        """Гребневая регрессия по активным столбцам. w0, prior — априорное среднее и сила
        (наследование параметров с прошлого шага: штраф prior*||w - w0||^2)."""
        idx = np.flatnonzero(active)
        w = np.zeros(self.K)
        if len(idx) == 0:
            return w
        As = self.A[:, idx]
        pr = np.broadcast_to(np.asarray(prior, float), (self.K,))[idx]   # prior: число или вектор длины K
        G = As.T @ As + np.diag(self.lam + pr)
        rhs = As.T @ self.b
        if w0 is not None:
            rhs = rhs + pr * np.asarray(w0)[idx]
        w[idx] = np.linalg.solve(G, rhs)
        return w

    def rss(self, w):
        r = self.b - self.A @ w
        return float(r @ r)

    def physical(self, w):
        """Коэффициенты в исходных единицах."""
        return w * self.sb / self.sA

    # --- доказательства «за» каждое ребро
    def evidence(self, active):
        """Для каждого монома j оценивает, что изменится, если его переключить.

        Активный моном удаляем, неактивный добавляем (формулы обновления решения
        без повторных подгонок). Возвращает словарь массивов длины K:
          rss   — доля необъяснённого текущей моделью (скаляр);
          gain  — относительное изменение RSS при переключении: (RSS_без - RSS_с)/RSS_с;
          t     — парный t-критерий по квадратам невязок на интервалах (критерий ЛР № 3);
          coef  — |w_j| у монома в модели «с ним»;
          tcoef — |w_j| / стандартная ошибка.
        """
        K, m = self.K, self.m
        act = np.asarray(active, bool)
        S, N = np.flatnonzero(act), np.flatnonzero(~act)
        floor = 1e-12
        if len(S):
            As = self.A[:, S]
            Ginv = np.linalg.inv(As.T @ As + self.lam * np.eye(len(S)))
            wS = Ginv @ (As.T @ self.b)
            r = self.b - As @ wS
        else:
            As = np.zeros((m, 0))
            Ginv = np.zeros((0, 0))
            wS = np.zeros(0)
            r = self.b.copy()
        rss = float(r @ r)
        gain, tst = np.zeros(K), np.zeros(K)
        coef, tcoef = np.zeros(K), np.zeros(K)
        r2 = r ** 2

        def paired_t(d):
            sd = d.std(axis=0, ddof=1) if m > 1 else np.ones(d.shape[1])
            return d.mean(axis=0) / (sd / np.sqrt(m) + 1e-30)

        if len(S):
            dg = np.diag(Ginv)
            Rn = r[:, None] + (As @ Ginv) * (wS / dg)[None, :]
            rn2 = (Rn ** 2).sum(axis=0)
            gain[S] = (rn2 - rss) / max(rss, floor)
            tst[S] = paired_t(Rn ** 2 - r2[:, None])
            sigma2 = rss / max(m - len(S), 1)
            coef[S] = np.abs(wS)
            tcoef[S] = np.abs(wS) / (np.sqrt(sigma2 * dg) + 1e-12)
        if len(N):
            AN = self.A[:, N]
            C = AN - As @ (Ginv @ (As.T @ AN)) if len(S) else AN
            cc = (C ** 2).sum(axis=0) + self.lam
            wj = (C.T @ r) / cc
            Ra = r[:, None] - C * wj[None, :]
            ra2 = (Ra ** 2).sum(axis=0)
            gain[N] = (rss - ra2) / np.maximum(ra2, floor)
            tst[N] = paired_t(r2[:, None] - Ra ** 2)
            sigma2 = ra2 / max(m - len(S) - 1, 1)
            coef[N] = np.abs(wj)
            tcoef[N] = np.abs(wj) / (np.sqrt(sigma2 / cc) + 1e-12)
        return dict(rss=rss, gain=gain, t=tst, coef=coef, tcoef=tcoef)

    def removable(self, active, rel_tol=0.10, t_crit=2.0):
        """Критерий удаления из ЛР № 3 (идущий от ЛР № 2): ребро можно убрать, если ошибка
        вырастет не более чем на rel_tol и парный t-критерий не значим (t < t_crit)."""
        ev = self.evidence(active)
        return np.asarray(active, bool) & (ev["t"] < t_crit) & (ev["gain"] <= rel_tol)

    # --- устойчивость по мини-батчам (идея ЛР № 1–2)
    def stability(self, rng, n_batches=16, frac=0.6, lam=1e-3):
        """|среднее| / σ коэффициента по мини-батчам интервалов (плотная модель)."""
        K, m = self.K, self.m
        k = max(int(round(frac * m)), 2)
        W = np.empty((n_batches, K))
        eye = np.eye(K)
        for bi in range(n_batches):
            idx = rng.choice(m, size=k, replace=False)
            As, bs = self.A[idx], self.b[idx]
            W[bi] = np.linalg.solve(As.T @ As + lam * eye, As.T @ bs)
        return np.abs(W.mean(axis=0)) / (W.std(axis=0) + 1e-9)
