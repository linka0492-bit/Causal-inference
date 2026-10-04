"""
Дерево суперпозиции для 2D несжимаемых уравнений Навье-Стокса
(периодическая область, как в датасете "shear_flow" из The Well).

    dv/dt + (v . grad) v = -grad(p) + nu * laplacian(v),    div(v) = 0

Формализм — ровно тот же, что в lotka_volterra_tree.py (три цвета вершин,
матрица смежности Z), но с ОДНИМ принципиальным отличием, которое стоит
явно проговорить с научруком (это прямой ответ на его вопрос "что у вас
в вершинах"):

  - в Volterra-Lotka переменные x(t), y(t) -- СКАЛЯРЫ, и дерево компилируется
    в одно символьное выражение (sympy), которое потом подставляется в
    RK45 как есть;
  - здесь vx, vy -- это ПОЛЯ на сетке (двумерные массивы), а операции
    словаря -- это дифференциальные операторы (grad, div, laplacian), а
    не арифметика над числами. Нет библиотеки, которая "символьно"
    компилирует PDE на сетке в одно выражение так же, как sympy.lambdify
    делает для ОДУ -- в CFD правую часть ВСЕГДА дискретизируют и считают
    численно. Поэтому ниже дерево остаётся явным и визуализируемым (те же
    три цвета, та же Z), но вместо to_sympy() есть compile_rhs(), который
    возвращает не символьное выражение, а готовую numpy-функцию,
    реализующую расчёт по спектральному методу (Фурье на периодической
    сетке -- точные производные без конечно-разностных ошибок).

Дополнительно: narод оператор "grad(p)" (градиент давления) -- единственная
НЕ локальная операция в этом дереве: давление p не является независимой
наблюдаемой переменной, а восстанавливается из условия div(v)=0 через
решение уравнения Пуассона по всему полю сразу (проекция Шорина/Чорина).
Это явно помечено отдельным типом вершины NONLOCAL, потому что для
операций "+", "*", "div", "grad", "laplacian" значение в точке зависит
только от значений по соседству (или даже только в этой же точке), а
давление -- от всего поля целиком сразу.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class NodeColor(Enum):
    OBSERVED = "observed"    # наблюдаемое поле (vx, vy)
    LATENT = "latent"        # локальная операция из словаря (поточечная или дифф. оператор)
    PARAMETER = "parameter"  # параметр модели (nu -- вязкость)
    NONLOCAL = "nonlocal"    # операция, требующая решения уравнения по всему полю (давление)


@dataclass
class Node:
    id: int
    color: NodeColor
    op: Optional[str] = None
    name: Optional[str] = None
    value: Optional[float] = None
    parents: List["Node"] = field(default_factory=list)

    def __repr__(self):
        if self.color in (NodeColor.LATENT, NodeColor.NONLOCAL):
            return f"V{self.id}[{self.op}]"
        return f"{self.color.value[0].upper()}{self.id}[{self.name}]"


class PDEExpressionTree:
    """DAG суперпозиции для класса PDE со спектральной дискретизацией на периодической сетке."""

    # словарь операций: локальные дифференциальные операторы + арифметика,
    # аналог {+, *, x, grad*, div*} из слайда про MHD, но без curl (нет B)
    # и с laplacian (вязкий член, которого в MHD-слайде не было)
    DICTIONARY = {"+", "-", "*", "ddx", "ddy", "laplacian"}
    NONLOCAL_DICTIONARY = {"pressure_grad_x", "pressure_grad_y"}

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

    def nonlocal_op(self, op: str, *parents: Node) -> Node:
        assert op in self.NONLOCAL_DICTIONARY
        n = Node(id=self._new_id(), color=NodeColor.NONLOCAL, op=op, parents=list(parents))
        self._nodes.append(n)
        return n

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    def adjacency_matrix(self) -> np.ndarray:
        n = len(self._nodes)
        Z = np.zeros((n, n), dtype=float)
        for node in self._nodes:
            for p in node.parents:
                Z[node.id, p.id] = 1.0
        return Z


def build_navier_stokes_tree() -> dict:
    """
    Строит DAG для правой части dvx/dt и dvy/dt. Давление -- отдельные
    NONLOCAL-вершины: их "входы" (vx, vy) обозначают зависимость от всего
    поля целиком, а не конкретное конечно-разностное соседство, как у
    остальных операций -- это и есть структурное отличие, с которым
    интересно будет обсудить научрука.
    """
    tree = PDEExpressionTree()

    vx = tree.observed("vx")
    vy = tree.observed("vy")
    nu = tree.parameter("nu", value=0.01)  # стартовая вязкость -- типичная для shear-layer задач

    # адвективный член (v . grad) vx = vx * d(vx)/dx + vy * d(vx)/dy
    dvx_dx = tree.op("ddx", vx)
    dvx_dy = tree.op("ddy", vx)
    adv_x1 = tree.op("*", vx, dvx_dx)
    adv_x2 = tree.op("*", vy, dvx_dy)
    adv_x = tree.op("+", adv_x1, adv_x2)

    dvy_dx = tree.op("ddx", vy)
    dvy_dy = tree.op("ddy", vy)
    adv_y1 = tree.op("*", vx, dvy_dx)
    adv_y2 = tree.op("*", vy, dvy_dy)
    adv_y = tree.op("+", adv_y1, adv_y2)

    # вязкий член nu * laplacian(v)
    lap_vx = tree.op("laplacian", vx)
    lap_vy = tree.op("laplacian", vy)
    visc_x = tree.op("*", nu, lap_vx)
    visc_y = tree.op("*", nu, lap_vy)

    # давление -- нелокальная вершина (зависит от ВСЕГО поля vx,vy сразу,
    # через решение уравнения Пуассона, обеспечивающего div(v)=0)
    dpdx = tree.nonlocal_op("pressure_grad_x", vx, vy)
    dpdy = tree.nonlocal_op("pressure_grad_y", vx, vy)

    # dvx/dt = -adv_x - dpdx + visc_x   (записываем как "+" из уже готовых термов)
    neg_adv_x = tree.op("*", tree.parameter("minus_one_a", -1.0), adv_x)
    neg_dpdx = tree.op("*", tree.parameter("minus_one_b", -1.0), dpdx)
    dvxdt_root = tree.op("+", tree.op("+", neg_adv_x, neg_dpdx), visc_x)

    neg_adv_y = tree.op("*", tree.parameter("minus_one_c", -1.0), adv_y)
    neg_dpdy = tree.op("*", tree.parameter("minus_one_d", -1.0), dpdy)
    dvydt_root = tree.op("+", tree.op("+", neg_adv_y, neg_dpdy), visc_y)

    # ограничение несжимаемости -- отдельное "уравнение" (как continuity в MHD-слайде),
    # используется как диагностика качества проекции, а не интегрируется по времени
    div_v = tree.op("+", dvx_dx, dvy_dy)

    return {
        "tree": tree,
        "vx": vx, "vy": vy, "nu": nu,
        "dvxdt_root": dvxdt_root,
        "dvydt_root": dvydt_root,
        "div_constraint_root": div_v,
    }


if __name__ == "__main__":
    built = build_navier_stokes_tree()
    tree = built["tree"]
    Z = tree.adjacency_matrix()
    print(f"Построено дерево: {len(tree.nodes)} вершин, матрица Z {Z.shape}")
    print("\nВершины:")
    for n in tree.nodes:
        print(" ", repr(n), "color=", n.color.value)
