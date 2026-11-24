"""Критерий прунинга (из ЛР № 2): «вершина выбрасывается, если её удаление ничего не меняет».

Для каждой вершины n дерева рассматриваются кандидаты замены поддерева:
  (a) n → 0   (вклад поддерева обнуляется),
  (b) n → один из его детей (удаляется «сестринская» часть).
Кандидат после упрощения и переподгонки параметров принимается, если
  1) сложность модели строго уменьшилась, и
  2) парный t-критерий по квадратам невязок на интервалах (e_cand² − e_full²) НЕ показывает значимого
     ухудшения:  t = mean(d)/(std(d)/√n) < t_crit,  и относительный рост ошибки не больше rel_tol.
Из принятых выбирается кандидат с наибольшим уменьшением сложности (при равенстве – с меньшей ошибкой);
процедура повторяется, пока есть что удалять.
"""
from __future__ import annotations
import numpy as np
from .expr import Model, Node, walk_with_parent, simplify, size
from .fit import IntegralProblem, fit_tree


def _paired_t(e_full, e_cand):
    d = e_cand ** 2 - e_full ** 2
    sd = d.std(ddof=1)
    if sd < 1e-12:
        return 0.0 if d.mean() <= 0 else np.inf
    return float(d.mean() / (sd / np.sqrt(len(d))))


def _replace(tree: Node, target_idx: int, repl: Node):
    """Копия дерева, где вершина с номером target_idx (в прямом обходе) заменена на repl."""
    t = tree.copy()
    if target_idx == 0:
        return repl.copy()
    for i, (n, p, ci) in enumerate(walk_with_parent(t)):
        if i == target_idx:
            p.ch[ci] = repl.copy()
            break
    return t


def prune_tree(tree: Node, prob: IntegralProblem, k: int, t_crit=2.0, rel_tol=0.10, max_steps=20):
    """Прунинг одного дерева. Возвращает (новое_дерево, список принятых удалений)."""
    cur = simplify(tree)
    fit_tree(cur, prob, k, max_nfev=40)
    log = []
    for _ in range(max_steps):
        try:
            e_full = prob.residual(cur, k)
        except Exception:
            break
        err_full = float(np.sum(e_full ** 2))
        nodes = list(walk_with_parent(cur))
        best = None
        for i, (n, p, ci) in enumerate(nodes):
            cands = [Node.const(0.0)]
            if n.kind == "op":
                for j, c in enumerate(n.ch):
                    other = n.ch[1 - j]
                    # не «схлопываем» коэффициент: произведение c·T → T (т.е. c := 1) не считается удалением
                    if n.op in "*/" and other.kind == "const":
                        continue
                    cands.append(c.copy())
            if n.kind == "const" and n.val == 0.0:
                continue
            for cd in cands:
                cand = simplify(_replace(cur, i, cd))
                if size(cand) >= size(cur):
                    continue
                fit_tree(cand, prob, k, max_nfev=40)
                try:
                    e_c = prob.residual(cand, k)
                except Exception:
                    continue
                err_c = float(np.sum(e_c ** 2))
                if not np.isfinite(err_c):
                    continue
                t = _paired_t(e_full, e_c)
                if t < t_crit and err_c <= err_full * (1 + rel_tol) + 1e-9:
                    gain = size(cur) - size(cand)
                    key = (-gain, err_c)
                    if best is None or key < best[0]:
                        best = (key, cand, i, t, gain)
        if best is None:
            break
        _, cur, i, t, gain = best
        log.append({"node_index": i, "t": t, "gain": gain})
    return cur, log


def prune_model(model: Model, prob: IntegralProblem, t_crit=2.0, rel_tol=0.10):
    """Прунинг обоих уравнений. Возвращает (новая_модель, лог)."""
    new, logs = [], []
    for k, t in enumerate(model.trees):
        nt, lg = prune_tree(t, prob, k, t_crit, rel_tol)
        new.append(nt); logs.append(lg)
    m = Model(new); m.meta = dict(model.meta)
    return m, logs
