"""Целевая функция выбора графов: точность + сложность + устойчивость к интервенциям.

    J(M) = w_acc · ln(acc) + w_cmp · C + w_stab · S                 (минимизируется)

acc   = среднее по двум уравнениям SSE/SST на приращениях (интегральное согласование; аналог 1 − R²);
C     = число вершин деревьев (операции + переменные + константы) + доп. штраф за параметры;
S     = ln(1 + instab/acc), где instab – средний рост ошибки на ИСХОДНЫХ данных после переподгонки модели
        на данных с интервенциями (шум, шум, зависящий от рыси, малая выборка, смещение). Заметим, что
        ln(acc) + S = ln(acc + instab): это логарифм ошибки на чистых данных, усреднённой по интервенциям.
        (Сам по себе относительный рост instab/acc «льстил» бы неточным моделям, поэтому берётся логарифм.)
Модели, вычислительно некорректные (деление на 0, inf), получают J = +inf.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .expr import Model
from .fit import IntegralProblem, fit_model


@dataclass
class Weights:
    w_acc: float = 1.0
    w_cmp: float = 0.03
    w_stab: float = 1.0
    param_cost: float = 0.5   # добавка к сложности за каждый параметр (в «вершинах»)
    acc_floor: float = 1e-3   # чтобы ln(acc) не уходил в −∞ при идеальной подгонке
    stab_skip_acc: float = 0.7  # устойчивость не считаем для заведомо плохих по точности моделей


@dataclass
class Scored:
    model: Model
    acc: float
    rel_x: float
    rel_y: float
    complexity: float
    n_nodes: int
    n_params: int
    instab: float
    J: float
    key: str
    gen: int = 0
    traj: dict = field(default_factory=dict)

    def summary(self):
        return (f"J={self.J:.3f} acc={self.acc:.4f} C={self.complexity:.1f} "
                f"S={self.instab:.3f}  {self.model}")


def stab_term(instab, acc, w: Weights):
    return float(np.log1p(instab / max(acc, w.acc_floor)))


def complexity(model: Model, w: Weights):
    return model.size() + w.param_cost * model.n_params


def score(model: Model, prob: IntegralProblem, stab, w: Weights, restarts=0, rng=None, max_nfev=60,
          gen=0, do_fit=True) -> Scored:
    """Подгонка параметров, вычисление точности, сложности, устойчивости и J."""
    if do_fit:
        rel_x, rel_y = fit_model(model, prob, restarts=restarts, rng=rng, max_nfev=max_nfev)
    else:
        rel_x, rel_y = (prob.rel_err(t, k) for k, t in enumerate(model.trees))
    C = complexity(model, w)
    if not (np.isfinite(rel_x) and np.isfinite(rel_y)):
        return Scored(model, np.inf, rel_x, rel_y, C, model.size(), model.n_params, np.inf, np.inf, model.key(), gen)
    acc = 0.5 * (rel_x + rel_y)
    instab = 0.0
    if stab is not None and acc < w.stab_skip_acc and w.w_stab > 0:
        instab, _ = stab.instability(model, prob, (rel_x, rel_y))
    S = stab_term(instab, acc, w)
    J = w.w_acc * np.log(max(acc, w.acc_floor)) + w.w_cmp * C + w.w_stab * S
    return Scored(model, acc, rel_x, rel_y, C, model.size(), model.n_params, instab, float(J), model.key(), gen)


def objective_terms(sc: Scored, w: Weights):
    """Разложение J на вклады (для графиков)."""
    S = stab_term(sc.instab, sc.acc, w)
    return {"accuracy": w.w_acc * np.log(max(sc.acc, w.acc_floor)),
            "complexity": w.w_cmp * sc.complexity,
            "stability": w.w_stab * S}
