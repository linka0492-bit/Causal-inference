"""Эксперимент: набор данных, разбиения, настройка гиперпараметров по J, проверки обобщения."""
import itertools

import numpy as np

from .baselines import BackwardLR3, DenseThreshold, STLSQ
from .forecaster import LinearGraphForecaster, Prepared
from .interventions import STABILITY_SET, corrupt
from .objective import EvalSet, evaluate
from .systems import build_collection


class Dataset:
    """Коллекция (выборка, модель): подготовленные выборки и их испорченные копии для обучения."""

    def __init__(self, n_per_system=40, seed=0, n_aug=2, systems=None):
        rng = np.random.default_rng(seed + 100)
        self.samples = build_collection(n_per_system, seed, systems)
        self.preps = [Prepared(s) for s in self.samples]
        self.aug = [[Prepared(corrupt(s, STABILITY_SET[rng.integers(len(STABILITY_SET))], rng))
                     for s in self.samples] for _ in range(n_aug)]

    def split(self, frac=(0.6, 0.2, 0.2), seed=0):
        """Случайное разбиение экземпляров внутри каждой системы: train / val / test (индексы)."""
        rng = np.random.default_rng(seed)
        out = [[], [], []]
        by = {}
        for i, s in enumerate(self.samples):
            by.setdefault(s.system, []).append(i)
        for ids in by.values():
            ids = rng.permutation(ids)
            a, b = int(frac[0] * len(ids)), int((frac[0] + frac[1]) * len(ids))
            out[0] += list(ids[:a]); out[1] += list(ids[a:b]); out[2] += list(ids[b:])
        return [sorted(o) for o in out]

    def train_preps(self, idx):
        return [self.preps[i] for i in idx] + [a[i] for a in self.aug for i in idx]

    def evalset(self, idx, seed=1, with_copies=True):
        return EvalSet([self.preps[i] for i in idx], seed=seed, with_copies=with_copies)


DEFAULT_GRID = dict(ridge=(0.1, 1.0, 10.0), tau=(0.4, 0.5, 0.6), max_flips=(2, 6), final_prune=(False, True))


def tune(train, valset, grid=DEFAULT_GRID, seed=0, verbose=True):
    """Перебор гиперпараметров; выбор по минимуму J на валидации. Возвращает (таблица, лучшая модель)."""
    rows, best = [], (np.inf, None)
    cache = None
    for ridge in grid["ridge"]:
        m = LinearGraphForecaster(ridge=ridge)
        m.fit(train, dagger_rounds=1, seed=seed, rows=cache)
        if cache is None:
            cache = m.rows_[:len(train) * 7]          # начальные состояния (без DAgger) переиспользуем
        for tau, k, fp in itertools.product(grid["tau"], grid["max_flips"], grid["final_prune"]):
            m.tau, m.max_flips, m.final_prune = tau, k, fp
            r = evaluate(m.predict, valset)
            row = dict(ridge=ridge, tau=tau, max_flips=k, final_prune=fp,
                       **{a: r[a] for a in ("J", "ln_acc", "C", "S", "acc", "exact", "f1")})
            rows.append(row)
            if r["J"] < best[0]:
                best = (r["J"], (ridge, tau, k, fp))
        if verbose:
            print(f"ridge={ridge}: лучший J пока {best[0]:.3f}")
    ridge, tau, k, fp = best[1]
    final = LinearGraphForecaster(ridge=ridge, tau=tau, max_flips=k, final_prune=fp)
    final.fit(train, dagger_rounds=1, seed=seed)
    return sorted(rows, key=lambda r: r["J"]), final


def fit_baselines(train):
    return {"линейная модель": None,
            "плотная регрессия + порог": DenseThreshold().fit(train),
            "STLSQ (SINDy)": STLSQ().fit(train),
            "критерий ЛР № 3 без обучения": BackwardLR3()}


def leave_out(ds, key, params, n_aug_use=2, verbose=True):
    """Обобщение на неизвестные системы: обучаемся без одной группы, проверяем на ней.
    key — 'system' или 'family'; params — гиперпараметры LinearGraphForecaster."""
    groups = sorted({getattr(s, key) for s in ds.samples})
    out = {}
    for g in groups:
        tr_idx = [i for i, s in enumerate(ds.samples) if getattr(s, key) != g]
        te_idx = [i for i, s in enumerate(ds.samples) if getattr(s, key) == g]
        m = LinearGraphForecaster(**params).fit(ds.train_preps(tr_idx), dagger_rounds=1)
        base = fit_baselines(ds.train_preps(tr_idx))
        es = ds.evalset(te_idx, with_copies=False)
        res = {"линейная модель": evaluate(m.predict, es)}
        for name, b in base.items():
            if b is not None:
                res[name] = evaluate(b.predict, es)
        out[g] = res
        if verbose:
            print(g, {k: round(v["exact"], 2) for k, v in res.items()})
    return out


def error_by_intervention(predict, samples, seed=7):
    """Ошибка графа (доля неверных рёбер) и доля точных структур при каждой интервенции в данные."""
    from .interventions import INTERVENTIONS
    from .objective import graph_error
    rng = np.random.default_rng(seed)
    out = {}
    for name in ["clean"] + list(INTERVENTIONS):
        errs, ex = [], []
        for s in samples:
            c = s if name == "clean" else INTERVENTIONS[name](s, rng)
            Zp = predict(Prepared(c))
            errs.append(graph_error(Zp, c.Z)); ex.append(bool((Zp == c.Z).all()))
        out[name] = dict(err=float(np.mean(errs)), exact=float(np.mean(ex)))
    return out


def param_error(forecast):
    """Относительная ошибка коэффициентов у верно угаданных рёбер (до уточнения по траектории)."""
    s = forecast.prep.s
    C = forecast.coef()
    hit = forecast.Z & s.Z
    if not hit.any():
        return np.nan
    return float(np.median(np.abs(C[hit] - s.W[hit]) / np.abs(s.W[hit])))
