"""Генетический алгоритм символьной регрессии для системы Лотки–Вольтерры.

Схема одного поколения (в духе указаний руководителя):
  1. Оценка популяции: подгонка параметров -> точность, сложность, устойчивость -> J.
  2. Лучшая модель поколения ПРУНИТСЯ критерием из ЛР № 2 (prune.py); результат возвращается в
     популяцию (элитизм) и записывается в «цепочку графов».
  3. Новое поколение: элита + потомки.  Потомки получаются
        а) турнирным отбором + кроссовер/мутация (классическая ветвь),
        б) кроссовером запрунённой лучшей модели («старое дерево») со СЛУЧАЙНО сгенерированной
           («новое дерево») + мутация (ветвь «порождение следующей модели»).
  4. Все оценённые структуры сохраняются в архив (выборка графов); оптимальный граф выбирается из архива.

Все ограничения априорного распределения (глубина, число вершин, корректность суперпозиции)
проверяются при порождении: некорректные деревья отбрасываются (равномерное распределение
на множестве допустимых деревьев).
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import numpy as np
from .expr import (Node, Model, random_tree, walk_with_parent, size, depth, simplify, consts,
                   rand_const, OPS, VARS)
from .fit import IntegralProblem
from .objective import Scored, Weights, score
from .prune import prune_model


@dataclass
class GAConfig:
    pop_size: int = 120
    n_gen: int = 60
    max_depth: int = 4          # глубина каждого дерева
    max_nodes: int = 13         # вершин в одном дереве
    init_depth: int = 3
    elite: int = 6
    tournament: int = 4
    p_cx: float = 0.6
    p_mut: float = 0.5
    frac_new_random: float = 0.15   # доля потомков «пруненое старое × новое случайное»
    prune_top: int = 3              # сколько лучших прунить в каждом поколении
    t_crit: float = 2.0
    rel_tol: float = 0.10
    seed: int = 0
    fresh_restarts: int = 2         # перезапуски подгонки для моделей со случайными константами
    max_tries: int = 30             # попыток породить корректного потомка
    variables: tuple = VARS
    ops: tuple = tuple(OPS)
    weights: Weights = field(default_factory=Weights)


@dataclass
class GAResult:
    config: GAConfig
    history: list           # по поколениям: статистики
    chain: list             # лучшая (запрунённая) модель каждого поколения: Scored
    archive: dict           # key -> Scored (лучший по J для структуры)
    n_evaluated: int
    best: Scored
    runtime: float
    sampled_chain: list = field(default_factory=list)   # лучший ПОРОЖДЁННЫЙ граф каждого поколения


class GeneticSR:
    def __init__(self, ds, scenarios_stab, cfg: GAConfig = None, verbose=True):
        from .interventions import StabilityEvaluator
        self.cfg = cfg or GAConfig()
        self.ds = ds
        self.prob = IntegralProblem(ds)
        self.stab = StabilityEvaluator(scenarios_stab) if scenarios_stab else None
        self.rng = np.random.default_rng(self.cfg.seed)
        self.verbose = verbose
        self.archive: dict[str, Scored] = {}
        self.n_eval = 0
        self.seen_keys = set()
        self.gen_log = {}
        # точки для проверки корректности суперпозиции (область определения)
        self.env_check = {"x": np.concatenate([self.prob.env["x"], ds.x]),
                          "y": np.concatenate([self.prob.env["y"], ds.y])}

    # ------------------------------------------------------------ ограничения
    def tree_ok(self, t: Node) -> bool:
        return size(t) <= self.cfg.max_nodes and depth(t) <= self.cfg.max_depth + 1

    def model_ok(self, m: Model) -> bool:
        if not all(self.tree_ok(t) for t in m.trees):
            return False
        return m.is_valid(self.env_check)

    # ------------------------------------------------------------ порождение
    def rand_tree(self, d=None):
        cfg = self.cfg
        d = d or int(self.rng.integers(2, cfg.init_depth + 1))
        method = "full" if self.rng.random() < 0.5 else "grow"
        return random_tree(self.rng, d, method, ops=cfg.ops, variables=cfg.variables)

    def random_model(self):
        for _ in range(self.cfg.max_tries * 3):
            m = Model([self.rand_tree(), self.rand_tree()])
            m = m.simplified()
            if self.model_ok(m) and m.size() >= 3:
                return m
        return Model([Node.var("x"), Node.var("y")])

    def _pick(self, tree: Node):
        nodes = list(walk_with_parent(tree))
        return nodes[int(self.rng.integers(len(nodes)))]

    def _replace_at(self, tree, parent, ci, new):
        if parent is None:
            return new
        parent.ch[ci] = new
        return tree

    def mutate_tree(self, tree: Node) -> Node:
        cfg, rng = self.cfg, self.rng
        t = tree.copy()
        r = rng.random()
        n, p, ci = self._pick(t)
        if r < 0.30:                     # точечная мутация
            if n.kind == "op":
                n.op = str(rng.choice([o for o in cfg.ops if o != n.op]))
            elif n.kind == "var":
                if rng.random() < 0.3:
                    n.kind, n.name, n.val = "const", None, rand_const(rng)
                else:
                    n.name = str(rng.choice(cfg.variables))
            else:
                if rng.random() < 0.2:
                    n.kind, n.val, n.name = "var", None, str(rng.choice(cfg.variables))
                else:
                    n.val *= float(np.exp(rng.normal(0, 0.4)))
        elif r < 0.60:                   # замена поддерева случайным
            sub = random_tree(rng, int(rng.integers(1, 3)), "grow", ops=cfg.ops, variables=cfg.variables)
            t = self._replace_at(t, p, ci, sub)
        elif r < 0.75:                   # «поднять» поддерево (hoist)
            t = n.copy()
        elif r < 0.88:                   # сжатие: операция -> один из детей
            ops_nodes = [(m, pp, cc) for m, pp, cc in walk_with_parent(t) if m.kind == "op"]
            if ops_nodes:
                m, pp, cc = ops_nodes[int(rng.integers(len(ops_nodes)))]
                t = self._replace_at(t, pp, cc, m.ch[int(rng.integers(2))].copy())
        else:                            # расширение: лист -> операция(лист, новый лист)
            leaves = [(m, pp, cc) for m, pp, cc in walk_with_parent(t) if m.kind != "op"]
            m, pp, cc = leaves[int(rng.integers(len(leaves)))]
            other = random_tree(rng, 0, "full", variables=cfg.variables)
            pair = [m.copy(), other]
            if rng.random() < 0.5:
                pair.reverse()
            t = self._replace_at(t, pp, cc, Node.binop(str(rng.choice(cfg.ops)), pair[0], pair[1]))
        return t

    def mutate(self, m: Model) -> Model:
        child = m.copy()
        k = int(self.rng.integers(2))
        child.trees[k] = self.mutate_tree(child.trees[k])
        if self.rng.random() < 0.15:     # иногда мутируем и второе уравнение
            j = 1 - k
            child.trees[j] = self.mutate_tree(child.trees[j])
        return child

    def crossover(self, a: Model, b: Model) -> Model:
        """Обмен случайными поддеревьями; уравнение выбирается случайно, иногда – с перекрёстным обменом."""
        child = a.copy()
        k = int(self.rng.integers(2))
        kb = k if self.rng.random() > 0.2 else 1 - k
        tb = b.trees[kb].copy()
        n1, p1, c1 = self._pick(child.trees[k])
        n2, _, _ = self._pick(tb)
        child.trees[k] = self._replace_at(child.trees[k], p1, c1, n2.copy())
        return child

    # ------------------------------------------------------------ оценка
    def evaluate(self, m: Model, gen, fresh=False, log=False) -> Scored | None:
        m = m.simplified()
        if not self.model_ok(m):
            return None
        rs = self.cfg.fresh_restarts if fresh else 0
        sc = score(m, self.prob, self.stab, self.cfg.weights, restarts=rs, rng=self.rng, gen=gen)
        self.n_eval += 1
        if not np.isfinite(sc.J):
            return None
        prev = self.archive.get(sc.key)
        is_new = prev is None
        if prev is None or sc.J < prev.J:
            self.archive[sc.key] = sc
        self.seen_keys.add(sc.key)
        if log and is_new:      # в цепочку порождённых графов попадают только новые структуры
            self.gen_log.setdefault(gen, []).append(sc)
        return sc

    def select(self, pop):
        idx = self.rng.integers(len(pop), size=self.cfg.tournament)
        return min((pop[i] for i in idx), key=lambda s: s.J)

    def child(self, pop, gen, parent_pruned: Scored | None):
        cfg = self.cfg
        for _ in range(cfg.max_tries):
            fresh = False
            if parent_pruned is not None and self.rng.random() < cfg.frac_new_random:
                # «старое запрунённое дерево» × «новое случайное» -> кроссовер -> мутация
                newm = self.random_model()
                c = self.crossover(parent_pruned.model, newm)
                c = self.mutate(c) if self.rng.random() < 0.7 else c
                fresh = True
            else:
                a = self.select(pop)
                if self.rng.random() < cfg.p_cx:
                    b = self.select(pop)
                    c = self.crossover(a.model, b.model)
                else:
                    c = a.model.copy()
                if self.rng.random() < cfg.p_mut or c.key() == a.model.key():
                    c = self.mutate(c)
            sc = self.evaluate(c, gen, fresh=fresh, log=True)
            if sc is not None:
                return sc
        return self.evaluate(self.random_model(), gen, fresh=True, log=True)

    # ------------------------------------------------------------ основной цикл
    def run(self) -> GAResult:
        cfg = self.cfg
        t0 = time.time()
        pop: list[Scored] = []
        guard = 0
        while len(pop) < cfg.pop_size and guard < cfg.pop_size * 20:
            guard += 1
            sc = self.evaluate(self.random_model(), 0, fresh=True, log=True)
            if sc is not None:
                pop.append(sc)
        history, chain = [], []
        pruned_best = None
        for g in range(cfg.n_gen):
            pop.sort(key=lambda s: s.J)
            # --- прунинг лучших, результат возвращается в популяцию
            pruned_list = []
            for s in pop[:cfg.prune_top]:
                pm, _ = prune_model(s.model, self.prob, cfg.t_crit, cfg.rel_tol)
                ps = self.evaluate(pm, g)
                if ps is not None:
                    pruned_list.append(ps)
            cands = pop[:1] + pruned_list
            best = min(cands, key=lambda s: s.J)
            pruned_best = best
            chain.append(best)
            pop = sorted(pop + pruned_list, key=lambda s: s.J)
            # дедупликация по структуре
            uniq, seen = [], set()
            for s in pop:
                if s.key not in seen:
                    uniq.append(s); seen.add(s.key)
            pop = uniq
            Js = np.array([s.J for s in pop])
            history.append({"gen": g, "best_J": float(best.J), "best_acc": float(best.acc),
                            "best_C": float(best.complexity), "best_S": float(np.log1p(best.instab / max(best.acc, 1e-3))),
                            "median_J": float(np.median(Js)), "n_unique": len(pop),
                            "n_archive": len(self.archive), "n_eval": self.n_eval,
                            "best_model": str(best.model)})
            if self.verbose and (g % 5 == 0 or g == cfg.n_gen - 1):
                print(f"[gen {g:3d}] J={best.J:7.3f} acc={best.acc:.4f} C={best.complexity:5.1f} "
                      f"S={best.instab:.3f} | uniq={len(pop):3d} archive={len(self.archive)} "
                      f"t={time.time() - t0:5.0f}s | {best.model}", flush=True)
            # --- следующее поколение
            nxt = pop[:cfg.elite]
            while len(nxt) < cfg.pop_size:
                nxt.append(self.child(pop, g + 1, pruned_best))
            pop = nxt
        best = min(self.archive.values(), key=lambda s: s.J)
        sampled = [min(self.gen_log[g], key=lambda s: s.J) for g in range(cfg.n_gen) if self.gen_log.get(g)]
        return GAResult(cfg, history, chain, self.archive, self.n_eval, best, time.time() - t0, sampled)
