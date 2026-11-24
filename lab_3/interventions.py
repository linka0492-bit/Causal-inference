"""Интервенции над МОДЕЛЬЮ и оценка устойчивости к интервенциям в ДАННЫЕ.

* StabilityEvaluator – устойчивость: модель переподгоняется на возмущённых данных, и измеряется,
  насколько вырастает ошибка на исходных данных (чем меньше, тем устойчивее).
* do_love – структурная интервенция «взаимная любовь» из ЛР № 2: do(u=1) добавляет к обоим уравнениям
  один и тот же член lam·u·x·y (к DAG добавляется подграф).
* intervention_table – расстояния между распределениями параметров (КС, W1/σ) до/после интервенции
  (как в ЛР № 2), но для найденной моделью.
"""
from __future__ import annotations
import numpy as np
from scipy import stats
from .expr import Model, Node
from .fit import IntegralProblem, fit_model, minibatch_params


class StabilityEvaluator:
    """Заранее строит интегральные задачи для набора возмущённых данных (детерминированно)."""

    def __init__(self, scenarios, max_nfev=15):
        self.scenarios = scenarios
        self.probs = [IntegralProblem(d) for d in scenarios]
        self.max_nfev = max_nfev

    def instability(self, model: Model, base_prob: IntegralProblem, base_rel):
        """Средний рост относительной ошибки на ИСХОДНЫХ данных после переподгонки на возмущённых.
        base_rel – (rel_x, rel_y) модели на исходных данных. Возвращает (среднее, список по сценариям)."""
        th0 = model.get_params()
        base = np.mean(base_rel)
        m = model.copy()
        deltas = []
        for pr in self.probs:
            m.set_params(th0)
            fit_model(m, pr, max_nfev=self.max_nfev)
            rel = [base_prob.rel_err(t, k) for k, t in enumerate(m.trees)]
            d = np.mean(rel) - base
            deltas.append(min(max(d, 0.0), 10.0) if np.isfinite(d) else 10.0)
        return float(np.mean(deltas)), deltas


def love_term(lam, u=1.0):
    """Подграф lam·u·x·y."""
    return Node.binop("*", Node.binop("*", Node.const(lam * u), Node.var("x")), Node.var("y"))


def do_love(model: Model, lam: float, u: float = 1.0) -> Model:
    """do(u): к обоим уравнениям добавляется lam·u·x·y (общий подграф x·y используется 3 потребителями)."""
    m = Model([Node.binop("+", t.copy(), love_term(lam, u)) for t in model.trees])
    m.meta = dict(model.meta)
    return m


def wasserstein_over_sigma(a, b):
    s = np.std(a)
    return float(stats.wasserstein_distance(a, b) / s) if s > 0 else np.nan


def intervention_table(model: Model, ds, scenarios: dict, n_batches=60, k=12, seed=0):
    """Для каждого сценария (имя → Dataset или (Dataset, размер батча)) и каждого параметра модели: KS D, p и W1/σ(base).
    Возвращает dict с массивами и подписями параметров."""
    base = minibatch_params(model, ds, n_batches, k, seed)
    # шум метода: base против независимого повтора base
    base2 = minibatch_params(model, ds, n_batches, k, seed + 1000)
    npar = base.shape[1]
    res = {"W1": {}, "D": {}, "p": {}, "n_params": npar}
    res["W1"]["__method_noise__"] = np.array([wasserstein_over_sigma(base[:, j], base2[:, j]) for j in range(npar)])
    res["D"]["__method_noise__"] = np.array([stats.ks_2samp(base[:, j], base2[:, j]).statistic for j in range(npar)])
    res["samples"] = {}
    for name, d in scenarios.items():
        kk = k
        if isinstance(d, tuple):          # (Dataset, размер батча) – например small_sample: батчи по 5 точек
            d, kk = d
        smp = minibatch_params(model, d, n_batches, min(kk, d.n), seed + 1)
        res["samples"][name] = smp
        res["W1"][name] = np.array([wasserstein_over_sigma(base[:, j], smp[:, j]) for j in range(npar)])
        ks = [stats.ks_2samp(base[:, j], smp[:, j]) for j in range(npar)]
        res["D"][name] = np.array([r.statistic for r in ks])
        res["p"][name] = np.array([r.pvalue for r in ks])
    res["base_samples"] = base
    return res
