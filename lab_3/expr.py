"""Дерево выражения (суперпозиция из {+, −, ×, ÷}) и модель как пара деревьев.

Модель Model = (f1, f2):  dx/dt = f1(x, y; θ),  dy/dt = f2(x, y; θ).
Листья дерева: переменные x, y (наблюдаемые вершины) и константы-параметры θ_k.
Внутренние вершины – операции (латентные вершины графа).
Дерево можно преобразовать в DAG (общие подвыражения без параметров, например x·y, объединяются),
а DAG – в бинарную матрицу инцидентности A (в обозначениях руководителя).
"""
from __future__ import annotations
import numpy as np

OPS = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
}
VARS = ("x", "y")
BIG = 1e7          # значения по модулю больше считаются «взрывом»
DEN_EPS = 1e-6     # |знаменатель| меньше → деление на ноль


class InvalidTree(Exception):
    """Дерево вычислительно некорректно (деление на ноль, inf/nan, взрыв значений)."""


class Node:
    __slots__ = ("kind", "op", "ch", "name", "val")

    def __init__(self, kind, op=None, ch=None, name=None, val=None):
        self.kind = kind          # 'op' | 'var' | 'const'
        self.op = op
        self.ch = ch or []
        self.name = name
        self.val = val

    # --- конструкторы
    @staticmethod
    def var(name): return Node("var", name=name)

    @staticmethod
    def const(v): return Node("const", val=float(v))

    @staticmethod
    def binop(op, a, b): return Node("op", op=op, ch=[a, b])

    def copy(self):
        return Node(self.kind, self.op, [c.copy() for c in self.ch], self.name, self.val)


# ----------------------------------------------------------------------------
# Обход
# ----------------------------------------------------------------------------
def walk(n: Node):
    """Все вершины в прямом порядке."""
    yield n
    for c in n.ch:
        yield from walk(c)


def walk_with_parent(n: Node, parent=None, idx=None):
    yield n, parent, idx
    for i, c in enumerate(n.ch):
        yield from walk_with_parent(c, n, i)


def size(n: Node) -> int:
    return 1 + sum(size(c) for c in n.ch)


def depth(n: Node) -> int:
    return 1 + (max(depth(c) for c in n.ch) if n.ch else 0)


def consts(n: Node):
    return [m for m in walk(n) if m.kind == "const"]


# ----------------------------------------------------------------------------
# Вычисление
# ----------------------------------------------------------------------------
def evaluate(n: Node, env: dict, check=True):
    """Векторизованное вычисление. env: {'x': array, 'y': array}. При check=True бросает InvalidTree."""
    if n.kind == "var":
        return env[n.name]
    if n.kind == "const":
        return np.full_like(env["x"], n.val)
    a = evaluate(n.ch[0], env, check)
    b = evaluate(n.ch[1], env, check)
    if n.op == "/":
        if check and np.min(np.abs(b)) < DEN_EPS:
            raise InvalidTree("division by ~0")
    with np.errstate(all="ignore"):
        r = OPS[n.op](a, b)
    if check and (not np.all(np.isfinite(r)) or np.max(np.abs(r)) > BIG):
        raise InvalidTree("non-finite or exploding")
    return r


def to_str(n: Node, digits=4) -> str:
    if n.kind == "var":
        return n.name
    if n.kind == "const":
        return f"{n.val:.{digits}g}"
    a, b = to_str(n.ch[0], digits), to_str(n.ch[1], digits)
    sym = {"*": "·", "/": "/", "+": "+", "-": "−"}[n.op]
    return f"({a} {sym} {b})"


def struct_key(n: Node) -> str:
    """Структура без значений констант (для кэша и архива)."""
    if n.kind == "var":
        return n.name
    if n.kind == "const":
        return "c"
    return f"({struct_key(n.ch[0])}{n.op}{struct_key(n.ch[1])})"


# ----------------------------------------------------------------------------
# Упрощение / каноническая форма
# ----------------------------------------------------------------------------
def _is_c(n, v=None):
    return n.kind == "const" and (v is None or n.val == v)


def _factors(n, out):
    """Развернуть вложенные умножения (и деление на константу) в список множителей."""
    if n.kind == "op" and n.op == "*":
        _factors(n.ch[0], out); _factors(n.ch[1], out)
    elif n.kind == "op" and n.op == "/" and _is_c(n.ch[1]) and n.ch[1].val != 0:
        _factors(n.ch[0], out); out.append(Node.const(1.0 / n.ch[1].val))
    else:
        out.append(n)


def simplify(n: Node) -> Node:
    """Алгебраическое упрощение: свёртка констант, нейтральные элементы, объединение констант
    в произведениях, канонический порядок множителей. Значение выражения сохраняется."""
    if n.kind != "op":
        return n.copy()
    a, b = simplify(n.ch[0]), simplify(n.ch[1])
    op = n.op
    if _is_c(a) and _is_c(b):
        with np.errstate(all="ignore"):
            v = float(OPS[op](a.val, b.val))
        return Node.const(v if np.isfinite(v) else 0.0)
    if op == "+":
        if _is_c(a, 0.0): return b
        if _is_c(b, 0.0): return a
    elif op == "-":
        if _is_c(b, 0.0): return a
        if struct_key(a) == struct_key(b) and not consts(a):
            return Node.const(0.0)
    elif op == "*":
        if _is_c(a, 0.0) or _is_c(b, 0.0): return Node.const(0.0)
    elif op == "/":
        if _is_c(a, 0.0): return Node.const(0.0)
        if _is_c(b, 1.0): return a
        if struct_key(a) == struct_key(b) and not consts(a):
            return Node.const(1.0)
    if op in "*/":
        fs = []
        _factors(Node.binop(op, a, b), fs)
        cs = [f for f in fs if f.kind == "const"]
        rest = [f for f in fs if f.kind != "const"]
        if op == "/" and not _is_c(b):
            # деление на непостоянное выражение: оставляем как есть (канонизируем только числитель)
            return Node.binop("/", a, b)
        c = 1.0
        for f in cs:
            c *= f.val
        if c == 0.0:
            return Node.const(0.0)
        rest.sort(key=lambda m: (m.kind != "var", struct_key(m)))
        parts = ([Node.const(c)] if (abs(c - 1.0) > 1e-15 or not rest or len(cs) > 0) else []) + rest
        if len(cs) == 0:
            parts = rest
        out = parts[0]
        for p in parts[1:]:
            out = Node.binop("*", out, p)
        return out
    return Node.binop(op, a, b)


# ----------------------------------------------------------------------------
# Случайная генерация
# ----------------------------------------------------------------------------
def rand_const(rng):
    return float(rng.choice([-1, 1]) * 10 ** rng.uniform(-2.0, 0.3))


def random_tree(rng, max_depth=4, method="grow", p_leaf=0.3, p_const=0.4, ops=tuple(OPS), variables=VARS):
    """Случайное дерево. method='full' – полное до max_depth, 'grow' – с досрочными листьями."""
    def leaf():
        if rng.random() < p_const:
            return Node.const(rand_const(rng))
        return Node.var(rng.choice(variables))

    def rec(d):
        if d >= max_depth or (method == "grow" and d > 0 and rng.random() < p_leaf):
            return leaf()
        return Node.binop(rng.choice(ops), rec(d + 1), rec(d + 1))
    return rec(0)


# ----------------------------------------------------------------------------
# Модель = пара деревьев
# ----------------------------------------------------------------------------
class Model:
    """dx/dt = trees[0](x,y;θ),  dy/dt = trees[1](x,y;θ)."""

    def __init__(self, trees):
        self.trees = list(trees)
        self.meta = {}

    def copy(self):
        m = Model([t.copy() for t in self.trees]); m.meta = dict(self.meta); return m

    # --- параметры (константы-листья) как единый вектор
    def const_nodes(self):
        return [c for t in self.trees for c in consts(t)]

    @property
    def n_params(self): return len(self.const_nodes())

    def get_params(self): return np.array([c.val for c in self.const_nodes()], float)

    def set_params(self, theta):
        for c, v in zip(self.const_nodes(), theta):
            c.val = float(v)

    def size(self): return sum(size(t) for t in self.trees)
    def depth(self): return max(depth(t) for t in self.trees)

    def simplified(self):
        m = Model([simplify(t) for t in self.trees]); m.meta = dict(self.meta); return m

    def key(self):
        return " | ".join(struct_key(t) for t in self.trees)

    def __str__(self):
        return f"dx/dt = {to_str(self.trees[0])};  dy/dt = {to_str(self.trees[1])}"

    def rhs(self, x, y, check=True):
        env = {"x": np.asarray(x, float), "y": np.asarray(y, float)}
        return tuple(evaluate(t, env, check) for t in self.trees)

    def is_valid(self, env_pts):
        """env_pts: dict x,y – массивы точек, на которых проверяется корректность."""
        try:
            for t in self.trees:
                evaluate(t, env_pts, True)
            return True
        except InvalidTree:
            return False

    # ------------------------------------------------------------------ граф
    def to_dag(self):
        """DAG модели с объединением общих подвыражений без параметров (например x·y).

        Возвращает dict: nodes (список словарей id,kind,label,layer), edges (список (src,dst)),
        outputs (id корней), и счётчики n_params, n_observed, n_latent.
        """
        nodes, edges, index = [], [], {}
        pc = [0]

        def add(key, kind, label):
            if key not in index:
                index[key] = len(nodes)
                nodes.append({"id": len(nodes), "kind": kind, "label": label, "key": key})
            return index[key]

        def rec(n):
            if n.kind == "var":
                return add(("v", n.name), "obs", n.name)
            if n.kind == "const":
                pc[0] += 1
                return add(("c", pc[0]), "param", f"{n.val:.3g}")
            a, b = rec(n.ch[0]), rec(n.ch[1])
            key = ("o", n.op, a, b)
            fresh = key not in index
            i = add(key, "op", {"*": "×", "/": "÷", "+": "+", "-": "−"}[n.op])
            if fresh:
                edges.append((a, i)); edges.append((b, i))
            return i

        outs = [rec(t) for t in self.trees]
        # слои для рисования: leaf=0, op = 1+max(слои детей)
        layer = {}
        for nd in nodes:
            if nd["kind"] != "op":
                layer[nd["id"]] = 0
        for nd in nodes:  # узлы добавлены в топологическом порядке (дети раньше родителей)
            if nd["kind"] == "op":
                srcs = [s for s, d in edges if d == nd["id"]]
                layer[nd["id"]] = 1 + max(layer[s] for s in srcs)
        for nd in nodes:
            nd["layer"] = layer[nd["id"]]
        n_obs = sum(nd["kind"] == "obs" for nd in nodes)
        n_par = sum(nd["kind"] == "param" for nd in nodes)
        n_lat = sum(nd["kind"] == "op" for nd in nodes)
        return {"nodes": nodes, "edges": edges, "outputs": outs,
                "n_params": n_par, "n_observed": n_obs, "n_latent": n_lat}

    def incidence_matrix(self):
        """Бинарная матрица инцидентности A (A[i,j]=1, если есть ребро i→j) и подписи вершин."""
        g = self.to_dag()
        n = len(g["nodes"])
        A = np.zeros((n, n), dtype=int)
        for s, d in g["edges"]:
            A[s, d] = 1
        return A, [nd["label"] for nd in g["nodes"]]

    def graph_counts(self):
        g = self.to_dag()
        return {"n_params": g["n_params"], "n_observed": g["n_observed"], "n_latent": g["n_latent"]}


def random_model(rng, max_depth=4, **kw):
    return Model([random_tree(rng, max_depth, **kw) for _ in range(2)])


def true_lv_model(alpha=0.481, beta=0.0248, gamma=0.926, delta=0.0275) -> Model:
    """Эталон: dx/dt = αx − βxy, dy/dt = δxy − γy."""
    X, Y = Node.var("x"), Node.var("y")
    C = Node.const
    f1 = Node.binop("-", Node.binop("*", C(alpha), X.copy()),
                    Node.binop("*", C(beta), Node.binop("*", X.copy(), Y.copy())))
    f2 = Node.binop("-", Node.binop("*", C(delta), Node.binop("*", X.copy(), Y.copy())),
                    Node.binop("*", C(gamma), Y.copy()))
    return Model([f1, f2])
