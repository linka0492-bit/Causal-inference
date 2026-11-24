"""Базовые методы выбора структуры на той же библиотеке и тех же признаках (для сравнения).

Все методы имеют интерфейс predict(prep) -> Z (bool, n_eq × K), как LinearGraphForecaster.
"""
import numpy as np


class DenseThreshold:
    """Плотная гребневая регрессия и порог на вкладе монома |w| (порог подбирается на обучении)."""

    def __init__(self, thr=0.1):
        self.thr = thr

    def _scores(self, prep):
        return prep.dense

    def fit(self, preps, grid=np.geomspace(1e-3, 1.0, 25)):
        best = (-1, self.thr)
        for thr in grid:
            tp = fp = fn = 0
            for p in preps:
                Z = self._pred(p, thr)
                tp += (Z & p.s.Z).sum(); fp += (Z & ~p.s.Z).sum(); fn += (~Z & p.s.Z).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            best = max(best, (f1, thr))
        self.thr = best[1]
        return self

    def _pred(self, prep, thr):
        Z = self._scores(prep) > thr
        for i in range(prep.n_eq):
            if not Z[i].any():
                Z[i, np.argmax(self._scores(prep)[i])] = True
        return Z

    def predict(self, prep):
        return self._pred(prep, self.thr)


class STLSQ(DenseThreshold):
    """Последовательная пороговая регрессия (SINDy): подгонка, отбрасывание малых вкладов, повтор."""

    def __init__(self, thr=0.1, iters=8):
        super().__init__(thr)
        self.iters = iters

    def _pred(self, prep, thr):
        Z = np.ones((prep.n_eq, prep.K), bool)
        for i, eq in enumerate(prep.eqs):
            act = np.ones(prep.K, bool)
            for _ in range(self.iters):
                w = np.abs(eq.fit(act))
                new = act & (w > thr)
                if not new.any():
                    new = np.zeros(prep.K, bool)
                    new[np.argmax(w)] = True
                if (new == act).all():
                    break
                act = new
            Z[i] = act
        return Z


class BackwardLR3:
    """Только критерий удаления из ЛР № 3, без обучения: от полного графа удаляем по одному
    самому слабому ребро, пока есть рёбра, удаление которых «ничего не меняет»."""

    def predict(self, prep):
        Z = prep.full()
        for _ in range(prep.n_eq * prep.K):
            cand = []
            for i, eq in enumerate(prep.eqs):
                if Z[i].sum() <= 1:
                    continue
                rem = eq.removable(Z[i])
                if rem.any():
                    ev = eq.evidence(Z[i])
                    j = np.flatnonzero(rem)[np.argmin(ev["gain"][rem])]
                    cand.append((ev["gain"][j], i, j))
            if not cand:
                break
            _, i, j = min(cand)
            Z[i, j] = False
        return Z
