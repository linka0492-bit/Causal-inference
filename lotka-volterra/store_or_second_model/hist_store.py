"""
Хранилище выборок параметров и байесовских гиперпараметров для РАЗНЫХ моделей.

Структура на диске (корень по умолчанию outputs/hist_store):
    <root>/<model>/<scenario>.csv    -- выборка параметров (строка = один мини-батч)
    <root>/<model>/<scenario>.json   -- имена параметров, мета, гиперпараметры
                                        (семейства norm/t/laplace/lognorm/gamma: params, AIC, KS-статистика)
    <root>/index.json                -- {model: [scenario, ...]} для быстрого обзора

Основные методы:
    put(model, scenario, names, samples, meta=None)  -- сохранить выборку и подогнать распределения
    get_samples(model, scenario) -> (names, ndarray)
    get_hyper(model, scenario)   -> dict
    models(), scenarios(model)
    compare(model, scen_a, scen_b, model_b=None) -> список метрик по параметрам
        (КС-тест с поправкой Холма, Вассерштейн, Йенсен-Шеннон, KL); model_b позволяет
        сравнивать сценарии из разных моделей, если у них одинаковые имена параметров.
"""

import json
from pathlib import Path

import numpy as np

import lv_common  # noqa: F401  (ставит заглушку sympy, если его нет)
from distribution_fits import fit_families
from distribution_distances import metrics, holm


class HistStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- запись ----------
    def put(self, model, scenario, names, samples, meta=None, fit=True):
        samples = np.asarray(samples, dtype=float)
        d = self.root / model
        d.mkdir(exist_ok=True)
        np.savetxt(d / f"{scenario}.csv", samples, delimiter=",", header=",".join(names), comments="")
        info = {"model": model, "scenario": scenario, "names": list(names),
                "n_batches": int(len(samples)), "meta": meta or {}, "params": {}}
        for j, nm in enumerate(names):
            s = samples[:, j]
            rec = {"mean": float(s.mean()), "std": float(s.std(ddof=1))}
            if fit:
                fits = fit_families(s)
                rec["fits"] = fits
                rec["best_by_aic"] = min(fits, key=lambda k: fits[k]["aic"]) if fits else None
            info["params"][nm] = rec
        with open(d / f"{scenario}.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        self._update_index()

    def _update_index(self):
        idx = {m: self.scenarios(m) for m in self.models()}
        with open(self.root / "index.json", "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)

    # ---------- чтение ----------
    def models(self):
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def scenarios(self, model):
        return sorted(p.stem for p in (self.root / model).glob("*.json"))

    def get_samples(self, model, scenario):
        d = self.root / model
        info = self.get_hyper(model, scenario)
        arr = np.loadtxt(d / f"{scenario}.csv", delimiter=",", skiprows=1)
        return info["names"], arr

    def get_hyper(self, model, scenario):
        with open(self.root / model / f"{scenario}.json", encoding="utf-8") as f:
            return json.load(f)

    # ---------- сравнение ----------
    def compare(self, model, scen_a, scen_b, model_b=None):
        model_b = model_b or model
        na, a = self.get_samples(model, scen_a)
        nb, b = self.get_samples(model_b, scen_b)
        assert na == nb, "у сравниваемых выборок должны совпадать имена параметров"
        rows = [metrics(a[:, j], b[:, j]) for j in range(len(na))]
        adj = holm(np.array([r["ks_p"] for r in rows]))
        for nm, r, p in zip(na, rows, adj):
            r["param"] = nm
            r["ks_p_holm"] = float(p)
        return rows
