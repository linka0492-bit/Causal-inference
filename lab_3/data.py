"""Данные Hudson Bay (заяц-беляк, канадская рысь, тыс. шкур) 1900–1920 и интервенции в данные.

Интервенции повторяют сценарии из ЛР № 2:
zero_lynx, noise, noise_cond, shuffle30, small_sample, add_shift, bias.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

YEARS = np.arange(1900, 1921, dtype=float)
HARE = np.array([30, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22, 25.4, 27.1,
                 40.3, 57, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7], float)
LYNX = np.array([4, 6.1, 9.8, 35.2, 59.4, 41.7, 19, 13, 8.3, 9.1, 7.4,
                 8, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6], float)


@dataclass
class Dataset:
    """Наблюдения двух переменных во времени (t – годы, x – заяц, y – рысь)."""
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    name: str = "base"
    meta: dict = field(default_factory=dict)

    def copy(self, name=None):
        return Dataset(self.t.copy(), self.x.copy(), self.y.copy(),
                       name or self.name, dict(self.meta))

    @property
    def n(self):
        return len(self.t)


def load_hudson_bay() -> Dataset:
    return Dataset(YEARS.copy(), HARE.copy(), LYNX.copy(), "base")


# ----------------------------------------------------------------------------
# Интервенции в данные
# ----------------------------------------------------------------------------
def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def zero_lynx(ds: Dataset, seed=0, level=1.0) -> Dataset:
    """Серия рыси заменена константой (level × среднее)."""
    d = ds.copy("zero_lynx")
    d.y = np.full_like(d.y, level * ds.y.mean())
    return d


def noise(ds: Dataset, seed=0, scale=0.3, bias=0.0) -> Dataset:
    """Аддитивный шум N(bias·σ, (scale·σ)²) в обе серии (σ – std серии)."""
    r = _rng(seed)
    d = ds.copy("noise" if bias == 0 else f"bias_{bias:+.2f}")
    d.x = np.clip(d.x + r.normal(bias * ds.x.std(), scale * ds.x.std(), d.n), 0.1, None)
    d.y = np.clip(d.y + r.normal(bias * ds.y.std(), scale * ds.y.std(), d.n), 0.1, None)
    return d


def noise_cond(ds: Dataset, seed=0, scale=0.3) -> Dataset:
    """Шум зайца пропорционален значению рыси."""
    r = _rng(seed)
    d = ds.copy("noise_cond")
    s = scale * ds.x.std() * ds.y / ds.y.mean()
    d.x = np.clip(d.x + r.normal(0, 1, d.n) * s, 0.1, None)
    return d


def shuffle(ds: Dataset, seed=0, frac=0.3) -> Dataset:
    """У доли frac моментов заяц и рысь поменяны местами."""
    r = _rng(seed)
    d = ds.copy(f"shuffle{int(frac * 100)}")
    idx = r.choice(d.n, int(round(frac * d.n)), replace=False)
    d.x[idx], d.y[idx] = ds.y[idx].copy(), ds.x[idx].copy()
    return d


def small_sample(ds: Dataset, seed=0, k=12) -> Dataset:
    """Оставлена случайная подвыборка из k моментов (порядок по времени сохранён)."""
    r = _rng(seed)
    idx = np.sort(r.choice(ds.n, k, replace=False))
    return Dataset(ds.t[idx], ds.x[idx], ds.y[idx], f"small_sample{k}", {"idx": idx})


def add_shift(ds: Dataset, model_pred, seed=0, n_new=5, hare_mult=1.5, resid_sd=None) -> Dataset:
    """Добавлены n_new точек в промежуточные моменты: значение модели + шум; заяц умножен на hare_mult.
    model_pred(t)->(x,y) – траектория подогнанной модели."""
    r = _rng(seed)
    tn = np.sort(r.choice(ds.t[:-1] + 0.5, n_new, replace=False))
    xp, yp = model_pred(tn)
    sdx = resid_sd[0] if resid_sd else 0.05 * ds.x.std()
    sdy = resid_sd[1] if resid_sd else 0.05 * ds.y.std()
    xn = np.clip(xp + r.normal(0, sdx, n_new), 0.1, None) * hare_mult
    yn = np.clip(yp + r.normal(0, sdy, n_new), 0.1, None)
    t = np.concatenate([ds.t, tn]); o = np.argsort(t)
    return Dataset(t[o], np.concatenate([ds.x, xn])[o], np.concatenate([ds.y, yn])[o],
                   f"add_shift{n_new}")


# Набор сценариев, по которым оценивается устойчивость (см. objective.py)
DEFAULT_SCENARIOS = [
    ("noise", lambda ds, s: noise(ds, s, 0.3)),
    ("noise_cond", lambda ds, s: noise_cond(ds, s, 0.3)),
    ("small_sample", lambda ds, s: small_sample(ds, s, 12)),
    ("bias_p025", lambda ds, s: noise(ds, s, 0.3, bias=0.25)),
]


def make_scenarios(ds: Dataset, seeds=(11, 22), scenarios=None):
    """Заранее сгенерировать фиксированный набор возмущённых наборов данных (детерминированно)."""
    out = []
    for name, fn in (scenarios or DEFAULT_SCENARIOS):
        for s in seeds:
            d = fn(ds, s)
            d.name = f"{name}#{s}"
            out.append(d)
    return out
