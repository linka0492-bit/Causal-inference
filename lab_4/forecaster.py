"""Линейная регрессионная модель прогноза структуры графа.

Идея. Структура модели — матрица Z (n_eq × K) включения мономов в уравнения. Прогнозная модель
fai: (выборка, текущий граф z_t) -> Γ, где Γ[i, j] — вероятность того, что ребро (i, j) входит в
целевой граф. Γ — линейная функция признаков ребра, общая для всех систем коллекции (обычная
гребневая регрессия по метке «ребро есть в ground truth»). Признаки ребра зависят от выборки и
текущего графа и не зависят от названий переменных, поэтому модель переносится между системами.

Прогноз — последовательность графов z_T -> z_{T-1} -> ... -> z_0:
  * z_T — полный граф (все мономы во всех уравнениях);
  * на каждом шаге модель считает Γ, ребро включается при Γ > tau, и за шаг меняется не больше
    max_flips самых уверенных рёбер (так получается видимая цепочка графов);
  * параметры подгоняются заново на каждом шаге, но наследуются от предыдущего (априорное
    среднее w_prev, как в -ln p(w|f) = (w - mu)^T A^-1 (w - mu)); у рёбер, которые остались,
    значения сохраняются, у вылетевших — исчезают, у новых — начинают с нуля;
  * в конце — необязательный прунинг по критерию из ЛР № 3 (если удаление ничего не меняет).

Критерий точности аппроксимации данных в обучение Γ не входит: Γ обучается только на
совпадении графа с эталоном. Данные участвуют в признаках (через критерий удаления/сохранения).
"""
from dataclasses import dataclass, field

import numpy as np

from .library import Equation, integral_design, term_structure

N_BASE = 14  # число базовых признаков ребра; в модель идут признаки × (ребро есть / ребра нет)


# ---------------------------------------------------------------- подготовка выборки
class Prepared:
    """Выборка, подготовленная к регрессии: интегральное согласование + статические признаки."""

    def __init__(self, sample, lam=1e-4, seed=0):
        self.s = sample
        self.terms = sample.terms
        self.n_eq, self.K = sample.n_eq, len(self.terms)
        A, B = integral_design(sample.t, sample.X, self.terms)
        self.eqs = [Equation(A, B[:, i], lam) for i in range(self.n_eq)]
        rng = np.random.default_rng(seed)
        self.stab = np.array([eq.stability(rng) for eq in self.eqs])           # критерий мини-батчей (ЛР № 2)
        dense = np.ones(self.K, bool)
        self.dense = np.array([np.abs(eq.fit(dense)) for eq in self.eqs])
        self.struct = np.array([[term_structure(e, i, self.n_eq) for e in self.terms]
                                for i in range(self.n_eq)])                       # (n_eq, K, 5)

    def full(self):
        return np.ones((self.n_eq, self.K), bool)


def edge_features(prep, Zc):
    """Признаки рёбер при текущем графе Zc -> массив (n_eq, K, N_BASE)."""
    Phi = np.zeros((prep.n_eq, prep.K, N_BASE))
    for i, eq in enumerate(prep.eqs):
        ev = eq.evidence(Zc[i])
        one = np.ones(prep.K)
        Phi[i, :, 0] = 1.0                                                       # свободный член
        Phi[i, :, 1] = np.clip(np.log10(np.maximum(ev["gain"], 0) + 1e-6), -6, 6) / 6
        Phi[i, :, 2] = np.clip(ev["t"], -5, 20) / 5                              # парный t (ЛР № 3)
        Phi[i, :, 3] = (np.clip(np.log10(ev["coef"] + 1e-6), -6, 2) + 6) / 8     # вклад монома
        Phi[i, :, 4] = np.clip(np.log10(ev["tcoef"] + 1e-3), -3, 3) / 3          # t-статистика коэффициента
        Phi[i, :, 5] = np.clip(prep.stab[i], 0, 5) / 5                           # устойчивость по батчам
        Phi[i, :, 6] = (np.clip(np.log10(prep.dense[i] + 1e-6), -6, 2) + 6) / 8  # вклад в плотной модели
        Phi[i, :, 7] = one * np.clip(np.log10(ev["rss"] + 1e-12), -12, 0) / 12   # качество текущей модели
        Phi[i, :, 8] = one * min(Zc[i].sum(), 10) / 10                           # размер уравнения
        Phi[i, :, 9:] = prep.struct[i]
    return Phi


def _expand(Phi, Zc):
    z = Zc[..., None].astype(float)
    return np.concatenate([Phi * z, Phi * (1.0 - z)], axis=-1)                   # веса отдельно для «есть/нет»


# ---------------------------------------------------------------- результат прогноза
@dataclass
class Step:
    Z: np.ndarray                    # граф после шага
    Gamma: np.ndarray                # вероятности, по которым сделан шаг (None для z_T)
    W: np.ndarray                    # нормированные параметры (n_eq, K)
    rss: list                        # доля необъяснённого по уравнениям
    flips: list = field(default_factory=list)   # [(i, j, +1 добавлено / -1 удалено)]
    kind: str = "regression"         # "start" | "regression" | "prune"

    @property
    def n_kept(self):
        return 0


@dataclass
class Forecast:
    prep: Prepared
    steps: list

    @property
    def Z(self):
        return self.steps[-1].Z

    def coef(self, step=-1):
        """Коэффициенты в исходных единицах (n_eq, K)."""
        W = self.steps[step].W
        return np.array([eq.physical(W[i]) for i, eq in enumerate(self.prep.eqs)])

    def formula(self, step=-1, digits=4):
        from .library import term_name
        Z, C = self.steps[step].Z, self.coef(step)
        names = self.prep.s.names
        out = []
        for i in range(self.prep.n_eq):
            parts = [f"{C[i, j]:+.{digits}g}*{term_name(e, names)}" for j, e in enumerate(self.prep.terms) if Z[i, j]]
            out.append(f"d{names[i]}/dt = " + (" ".join(parts) if parts else "0"))
        return out


# ---------------------------------------------------------------- модель
class LinearGraphForecaster:
    def __init__(self, ridge=1.0, tau=0.5, max_flips=3, max_steps=40, final_prune=True,
                 prior=1e-3, class_power=0.5):
        self.ridge, self.tau, self.max_flips = ridge, tau, max_flips
        self.max_steps, self.final_prune, self.prior = max_steps, final_prune, prior
        self.class_power = class_power
        self.coef_ = None

    # --- Γ
    def gamma(self, prep, Zc):
        Phi = _expand(edge_features(prep, Zc), Zc)
        return np.clip(Phi @ self.coef_, 0.0, 1.0)

    # --- обучение
    @staticmethod
    def corrupted_states(prep, rng, n):
        """Прямой процесс q(z_t | z_0): к эталону добавляются ложные рёбра, иногда теряется истинное."""
        Z0 = prep.s.Z
        states = [prep.full()]
        false_idx = np.argwhere(~Z0)
        for _ in range(n):
            Zc = Z0.copy()
            q = int(rng.choice([0, 1, 2, 3, 5, 8]))
            if q and len(false_idx):
                pick = rng.choice(len(false_idx), size=min(q, len(false_idx)), replace=False)
                Zc[tuple(false_idx[pick].T)] = True
            if rng.random() < 0.3 and Z0.sum() > 1:
                i, j = np.argwhere(Z0)[rng.integers(Z0.sum())]
                Zc[i, j] = False
            states.append(Zc)
        return states

    def _rows(self, prep, Zc):
        P = _expand(edge_features(prep, Zc), Zc).reshape(-1, 2 * N_BASE)
        y = prep.s.Z.astype(float).ravel()
        return P, y, np.full(len(y), 1.0 / len(y))                  # каждая (выборка, состояние) — равный вес

    def _solve(self, rows):
        P = np.vstack([r[0] for r in rows])
        y = np.concatenate([r[1] for r in rows])
        w = np.concatenate([r[2] for r in rows])
        pos = (y * w).sum() / w.sum()
        cw = np.where(y > 0.5, ((1 - pos) / max(pos, 1e-3)) ** self.class_power, 1.0)   # баланс классов
        sw = w * cw
        G = P.T @ (P * sw[:, None]) + self.ridge * np.eye(P.shape[1])
        self.coef_ = np.linalg.solve(G, P.T @ (sw * y))
        return self

    def fit(self, preps, n_corrupt=6, dagger_rounds=1, seed=0, rows=None):
        """preps — список Prepared с эталонами. dagger_rounds — сколько раз дообучать на графах,
        которые модель посещает сама (чтобы не зависеть только от искусственных искажений)."""
        rng = np.random.default_rng(seed)
        if rows is None:
            rows = [self._rows(p, Zc) for p in preps for Zc in self.corrupted_states(p, rng, n_corrupt)]
        self.rows_ = list(rows)
        self._solve(self.rows_)
        for _ in range(dagger_rounds):
            new = []
            for p in preps:
                fc = self.forecast(p, final_prune=False)
                new += [self._rows(p, st.Z) for st in fc.steps[:-1]]
            self.rows_ += new
            self._solve(self.rows_)
        return self

    # --- прогноз
    def _refit(self, prep, Zc, Wprev):
        W = np.zeros((prep.n_eq, prep.K))
        for i, eq in enumerate(prep.eqs):
            kept = Zc[i] & (Wprev[i] != 0) if Wprev is not None else np.zeros(prep.K, bool)
            W[i] = eq.fit(Zc[i], w0=Wprev[i] if Wprev is not None else None, prior=self.prior * kept)
        return W

    def _step(self, prep, Zc, W, Znew, gamma, flips, kind):
        W2 = self._refit(prep, Znew, W)
        rss = [eq.rss(W2[i]) for i, eq in enumerate(prep.eqs)]
        return Step(Znew, gamma, W2, rss, flips, kind)

    def forecast(self, prep, final_prune=None, tau=None, max_flips=None):
        if not isinstance(prep, Prepared):
            prep = Prepared(prep)
        tau = self.tau if tau is None else tau
        max_flips = self.max_flips if max_flips is None else max_flips
        final_prune = self.final_prune if final_prune is None else final_prune
        Zc = prep.full()
        W = self._refit(prep, Zc, None)
        steps = [Step(Zc, None, W, [eq.rss(W[i]) for i, eq in enumerate(prep.eqs)], [], "start")]
        for _ in range(self.max_steps):
            G = self.gamma(prep, Zc)
            diff = (G > tau) != Zc
            if not diff.any():
                break
            margin = np.where(diff, np.abs(G - tau), -1.0)
            order = np.argsort(-margin, axis=None)[:min(max_flips, int(diff.sum()))]
            Znew, flips = Zc.copy(), []
            for o in order:
                i, j = np.unravel_index(o, Zc.shape)
                if Zc[i, j] and Znew[i].sum() == 1:
                    continue                                          # в уравнении должен остаться хоть один член
                Znew[i, j] = not Zc[i, j]
                flips.append((int(i), int(j), 1 if Znew[i, j] else -1))
            if not flips:
                break
            st = self._step(prep, Zc, W, Znew, G, flips, "regression")
            steps.append(st)
            Zc, W = st.Z, st.W
        if final_prune:
            for _ in range(prep.n_eq * prep.K):
                cand = []
                for i, eq in enumerate(prep.eqs):
                    rem = eq.removable(Zc[i]) if Zc[i].sum() > 1 else np.zeros(prep.K, bool)
                    if rem.any():
                        ev = eq.evidence(Zc[i])
                        j = np.flatnonzero(rem)[np.argmin(ev["gain"][rem])]
                        cand.append((ev["gain"][j], i, j))
                if not cand:
                    break
                _, i, j = min(cand)
                Znew = Zc.copy()
                Znew[i, j] = False
                st = self._step(prep, Zc, W, Znew, None, [(int(i), int(j), -1)], "prune")
                steps.append(st)
                Zc, W = st.Z, st.W
        return Forecast(prep, steps)

    def predict(self, prep):
        return self.forecast(prep).Z

    # --- сохранение: вся модель — вектор из 2*N_BASE весов и пять гиперпараметров
    def save(self, path):
        import json
        json.dump(dict(coef=self.coef_.tolist(), ridge=self.ridge, tau=self.tau, max_flips=self.max_flips,
                       final_prune=self.final_prune, prior=self.prior), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        import json
        d = json.load(open(path))
        m = cls(ridge=d["ridge"], tau=d["tau"], max_flips=d["max_flips"], final_prune=d["final_prune"], prior=d["prior"])
        m.coef_ = np.array(d["coef"])
        return m
