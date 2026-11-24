"""Метрики прогноза графа и целевая функция J = ln(acc) + c*C + S (по образцу ЛР № 3).

acc — ошибка восстановления графа (а не ошибка по данным): (FP + FN) / |эталон|, плюс eps.
C   — сложность спрогнозированных моделей: для каждого ребра степень монома - 1 умножений,
      +1 умножение на коэффициент, +1 сложение, +0.5 за параметр (константа без умножения).
S   — неустойчивость к интервенциям в данные: S = ln(1 + Δ/acc), Δ — средний рост ошибки
      на 8 испорченных копиях выборки (4 вида порчи × 2 розыгрыша, как в ЛР № 3).
"""
import numpy as np

from .forecaster import Prepared
from .interventions import stability_copies
from .library import make_library

EPS = 0.02


def edge_counts(Zp, Zt):
    return int((Zp & Zt).sum()), int((Zp & ~Zt).sum()), int((~Zp & Zt).sum())


def graph_error(Zp, Zt):
    tp, fp, fn = edge_counts(Zp, Zt)
    return (fp + fn) / max(int(Zt.sum()), 1)


def complexity(Z, n_vars):
    deg = np.array([sum(e) for e in make_library(n_vars)])
    cost = np.where(deg == 0, 1.5, deg - 1 + 2.5)          # константа: параметр + сумма
    return float((Z * cost[None, :]).sum())


class EvalSet:
    """Выборки для оценки + их испорченные копии (готовятся один раз)."""

    def __init__(self, samples, seed=0, draws=2, with_copies=True):
        rng = np.random.default_rng(seed)
        self.items = []
        for s in samples:
            clean = s if isinstance(s, Prepared) else Prepared(s)
            copies = [Prepared(c) for c in stability_copies(clean.s, rng, draws)] if with_copies else []
            self.items.append((clean, copies))


def evaluate(predict, evalset, c=0.03, eps=EPS):
    """predict(prep) -> Z (bool). Возвращает словарь метрик и составляющие J."""
    errs, dels, cs, exact, tp, fp, fn = [], [], [], [], 0, 0, 0
    per_sample = []
    for clean, copies in evalset.items:
        Zp = predict(clean)
        e = graph_error(Zp, clean.s.Z)
        a, b, d = edge_counts(Zp, clean.s.Z)
        tp, fp, fn = tp + a, fp + b, fn + d
        errs.append(e)
        cs.append(complexity(Zp, clean.s.n_vars))
        exact.append(bool((Zp == clean.s.Z).all()))
        if copies:
            ec = np.mean([graph_error(predict(cp), cp.s.Z) for cp in copies])
            dels.append(max(ec - e, 0.0))
        per_sample.append(dict(system=clean.s.system, err=e, exact=exact[-1]))
    acc = float(np.mean(errs)) + eps
    C = float(np.mean(cs))
    delta = float(np.mean(dels)) if dels else 0.0
    S = float(np.log(1 + delta / acc))
    J = float(np.log(acc) + c * C + S)
    return dict(J=J, ln_acc=float(np.log(acc)), C=C, S=S, acc=acc - eps, delta=delta,
                exact=float(np.mean(exact)), f1=2 * tp / max(2 * tp + fp + fn, 1),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), per_sample=per_sample)
