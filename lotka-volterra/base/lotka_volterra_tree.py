"""
Экспрессионные деревья (суперпозиции) для класса уравнений Volterra-Lotka.

Формализм соответствует схеме из слайдов по MHD и Ван-дер-Полю:
  - три цвета вершин: наблюдаемая переменная (OBSERVED), промежуточная
    операция из словаря (LATENT), параметр модели (PARAMETER);
  - структура графа задаётся матрицей смежности Z (здесь строится бинарная
    версия; релаксированная непрерывная версия P(G) строится поверх неё
    на следующем шаге, заменой единиц на вероятности присутствия ребра);
  - это DAG, а не строго дерево: одинаковые поддеревья (например, x*y)
    могут переиспользоваться несколькими родителями.
"""

import numpy as np
import sympy as sp
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
from scipy.integrate import solve_ivp


class NodeColor(Enum):
    OBSERVED = "observed"      # наблюдаемая переменная (X_i)
    LATENT = "latent"          # промежуточная операция (V_i) из словаря операций
    PARAMETER = "parameter"    # параметр модели (theta)


@dataclass
class Node:
    id: int
    color: NodeColor
    op: Optional[str] = None            # имя операции из словаря {+, -, *, /}, для LATENT
    name: Optional[str] = None          # имя переменной/параметра, для OBSERVED/PARAMETER
    value: Optional[float] = None       # текущее числовое значение параметра (для PARAMETER)
    parents: List["Node"] = field(default_factory=list)  # входы операции

    def __repr__(self):
        if self.color == NodeColor.LATENT:
            return f"V{self.id}[{self.op}]"
        return f"{self.color.value[0].upper()}{self.id}[{self.name}]"


class ExpressionTree:
    """DAG суперпозиции, задающий одну скалярную правую часть уравнения ОДУ."""

    DICTIONARY = {"+", "-", "*", "/"}  # словарь допустимых операций для этого класса уравнений

    def __init__(self):
        self._nodes: List[Node] = []
        self._counter = 0

    def _new_id(self) -> int:
        i = self._counter
        self._counter += 1
        return i

    def observed(self, name: str) -> Node:
        n = Node(id=self._new_id(), color=NodeColor.OBSERVED, name=name)
        self._nodes.append(n)
        return n

    def parameter(self, name: str, value: float) -> Node:
        n = Node(id=self._new_id(), color=NodeColor.PARAMETER, name=name, value=value)
        self._nodes.append(n)
        return n

    def op(self, op: str, *parents: Node) -> Node:
        assert op in self.DICTIONARY, f"операция {op} отсутствует в словаре {self.DICTIONARY}"
        n = Node(id=self._new_id(), color=NodeColor.LATENT, op=op, parents=list(parents))
        self._nodes.append(n)
        return n

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    def adjacency_matrix(self) -> np.ndarray:
        """
        Бинарная матрица смежности Z (n x n): Z[i, j] = 1, если узел j —
        вход узла i. В релаксированной версии P(G) единицы заменяются
        на непрерывные вероятности присутствия соответствующего ребра.
        """
        n = len(self._nodes)
        Z = np.zeros((n, n), dtype=float)
        for node in self._nodes:
            for p in node.parents:
                Z[node.id, p.id] = 1.0
        return Z

    def to_sympy(self, root: Node):
        """Рекурсивно компилирует дерево, начиная с root, в выражение SymPy."""
        cache = {}

        def visit(node: Node):
            if node.id in cache:
                return cache[node.id]
            if node.color in (NodeColor.OBSERVED, NodeColor.PARAMETER):
                expr = sp.Symbol(node.name)
            else:
                args = [visit(p) for p in node.parents]
                if node.op == "+":
                    expr = sp.Add(*args)
                elif node.op == "-":
                    expr = args[0] - sum(args[1:])
                elif node.op == "*":
                    expr = sp.Mul(*args)
                elif node.op == "/":
                    expr = args[0] / args[1]
                else:
                    raise ValueError(node.op)
            cache[node.id] = expr
            return expr

        return visit(root)


def build_lotka_volterra() -> dict:
    """
    Строит два дерева (общий DAG с переиспользуемым поддеревом x*y) для системы:

        dx/dt = alpha * x - beta * x * y
        dy/dt = delta * x * y - gamma * y

    Наблюдаемые переменные: x (жертва), y (хищник).
    Параметры: alpha, beta, gamma, delta — стартовые значения для шага 3
    (типовые порядки величин из литературы по Hudson Bay lynx-hare,
    используются как инициализация перед последующей подгонкой).
    """
    tree = ExpressionTree()

    x = tree.observed("x")
    y = tree.observed("y")

    alpha = tree.parameter("alpha", value=0.55)
    beta = tree.parameter("beta", value=0.028)
    gamma = tree.parameter("gamma", value=0.84)
    delta = tree.parameter("delta", value=0.026)

    # dx/dt = alpha*x - beta*x*y
    xy = tree.op("*", x, y)
    term1 = tree.op("*", alpha, x)
    term2 = tree.op("*", beta, xy)
    dxdt_root = tree.op("-", term1, term2)

    # dy/dt = delta*x*y - gamma*y   (узел xy переиспользуется — это DAG, не дерево)
    term3 = tree.op("*", delta, xy)
    term4 = tree.op("*", gamma, y)
    dydt_root = tree.op("-", term3, term4)

    return {
        "tree": tree,
        "x": x, "y": y,
        "params": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        "dxdt_root": dxdt_root,
        "dydt_root": dydt_root,
    }


def demo():
    built = build_lotka_volterra()
    tree: ExpressionTree = built["tree"]

    dxdt_expr = tree.to_sympy(built["dxdt_root"])
    dydt_expr = tree.to_sympy(built["dydt_root"])

    print("Символьный вид уравнений, скомпилированных из дерева:")
    print("  dx/dt =", sp.simplify(dxdt_expr))
    print("  dy/dt =", sp.simplify(dydt_expr))

    Z = tree.adjacency_matrix()
    print("\nМатрица смежности Z (бинарная версия), размер", Z.shape)
    print(Z.astype(int))

    print("\nВершины дерева/DAG и их цвета:")
    for n in tree.nodes:
        print(" ", repr(n), "color=", n.color.value)

    # Численная проверка: lambdify + solve_ivp, убеждаемся, что дерево
    # действительно задаёт корректную правую часть системы ОДУ.
    x_s, y_s = sp.symbols("x y")
    subs_params = {
        sp.Symbol(name): node.value for name, node in built["params"].items()
    }

    dxdt_num = sp.lambdify((x_s, y_s), dxdt_expr.subs(subs_params), "numpy")
    dydt_num = sp.lambdify((x_s, y_s), dydt_expr.subs(subs_params), "numpy")

    def rhs(t, state):
        x_val, y_val = state
        return [dxdt_num(x_val, y_val), dydt_num(x_val, y_val)]

    sol = solve_ivp(rhs, t_span=(0, 50), y0=[10.0, 5.0], t_eval=np.linspace(0, 50, 200))

    print("\nПроверка интегрированием (первые 5 точек x(t), y(t)):")
    for t_val, x_val, y_val in zip(sol.t[:5], sol.y[0][:5], sol.y[1][:5]):
        print(f"  t={t_val:6.2f}  x={x_val:8.4f}  y={y_val:8.4f}")


if __name__ == "__main__":
    demo()
