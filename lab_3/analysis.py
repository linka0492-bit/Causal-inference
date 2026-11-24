"""Анализ результатов ГА: символьное раскрытие модели, проверка эквивалентности Лотке–Вольтерре,
уточнение лучших моделей по траектории ОДУ, траектории цепочки графов, сравнение при интервенции do(u=1)."""
from __future__ import annotations
import numpy as np
import sympy as sp
from .expr import Node, Model
from .fit import fit_trajectory, simulate
from .interventions import do_love

X_, Y_ = sp.symbols("x y")


def to_sympy(n: Node):
    if n.kind == "var":
        return X_ if n.name == "x" else Y_
    if n.kind == "const":
        return sp.Float(n.val)
    a, b = to_sympy(n.ch[0]), to_sympy(n.ch[1])
    return {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[n.op]


def expanded(model: Model, digits=4):
    """Раскрытые скобки: (dx/dt, dy/dt) как sympy-выражения с округлёнными коэффициентами."""
    out = []
    for t in model.trees:
        e = sp.expand(sp.cancel(sp.together(to_sympy(t))))
        e = e.xreplace({a: sp.Float(round(float(a), digits)) for a in e.atoms(sp.Float)})
        out.append(e)
    return tuple(out)


def expanded_str(model: Model, digits=4):
    ex, ey = expanded(model, digits)
    return f"dx/dt = {sp.sstr(ex)};  dy/dt = {sp.sstr(ey)}"


def poly_terms(expr):
    """{(i,j): coef} для многочлена от x,y или None, если выражение не полиномиально."""
    try:
        p = sp.Poly(expr, X_, Y_)
    except Exception:
        return None
    return {m: float(c) for m, c in p.terms()}


LV_KEY = "+x1y0,-x1y1 | -x0y1,+x1y1"


def matches_lv(model: Model):
    """Раскрытая модель в классе эквивалентности Лотки–Вольтерры: dx/dt = a·x − b·x·y, dy/dt = d·x·y − g·y
    (существенные на данных мономы – по критерию equiv_key)."""
    return equiv_key(model) == LV_KEY


def lv_coefficients(model: Model):
    """(α, β, δ, γ) из раскрытой модели, если она в классе ЛВ (малые несущественные члены отбрасываются)."""
    if not matches_lv(model):
        return None
    ex, ey = expanded(model, 8)
    tx, ty = poly_terms(ex), poly_terms(ey)
    return (tx[(1, 0)], -tx[(1, 1)], ty[(1, 1)], -ty[(0, 1)])


# ----------------------------------------------------------------------------
def trajectory_info(model: Model, ds, max_nfev=40, tfine=None):
    """Подгонка по траектории (на копии) и расчёт RSS и гладкой траектории."""
    m = model.copy()
    info = fit_trajectory(m, ds, fit_init=True, max_nfev=max_nfev)
    tf = np.linspace(ds.t[0], ds.t[-1], 400) if tfine is None else tfine
    sim = simulate(m, info["x0"], info["y0"], tf) if np.isfinite(info["rss"]) else None
    info.update({"t": tf, "sim": sim, "model": m})
    return info


def refine_top(result, ds, topk=10, w=None, reps=None):
    """Для top-K архива по J: подгонка траектории; итоговый балл J_final, где точность – RSS/SST
    (по траектории), сложность и устойчивость – как в J. Возвращает список dict, отсортированный."""
    from .objective import Weights
    w = w or result.config.weights
    sst = float(np.sum((ds.x - ds.x.mean()) ** 2) + np.sum((ds.y - ds.y.mean()) ** 2))
    cand = (reps if reps is not None else sorted(result.archive.values(), key=lambda s: s.J))[:topk]
    rows = []
    for sc in cand:
        info = trajectory_info(sc.model, ds)
        rel = info["rss"] / sst if np.isfinite(info["rss"]) else np.inf
        S = float(np.log1p(sc.instab / max(sc.acc, w.acc_floor)))
        Jf = (w.w_acc * np.log(max(rel, w.acc_floor)) + w.w_cmp * sc.complexity + w.w_stab * S) if np.isfinite(rel) else np.inf
        rows.append({"scored": sc, "rss": info["rss"], "rel_traj": rel, "J_final": float(Jf), "info": info,
                     "lv_equiv": matches_lv(sc.model), "expanded": expanded_str(sc.model)})
    rows.sort(key=lambda r: r["J_final"])
    return rows


def chain_trajectories(chain, ds, max_nfev=30):
    """Траектории для каждой модели цепочки (список Scored; мемоизация по структуре и параметрам)."""
    chain = getattr(chain, "chain", chain)
    memo, out = {}, []
    for sc in chain:
        k = (sc.key, tuple(np.round(sc.model.get_params(), 6)))
        if k not in memo:
            memo[k] = trajectory_info(sc.model, ds, max_nfev=max_nfev)
        out.append(memo[k])
    return out


def simulate_do_love(model: Model, x0, y0, t_eval, lam, t_do=1910.0):
    """Траектория с интервенцией do(u=1) c момента t_do: до t_do – исходная модель, после – модель с
    добавленным членом lam·u·x·y. Возвращает (времена, состояния) для t >= t_do или (None, None)."""
    t_eval = np.asarray(t_eval, float)
    pre = np.append(t_eval[t_eval < t_do], t_do)
    a = simulate(model, x0, y0, pre)
    if a is None:
        return None, None
    s0 = a[-1]
    post = np.concatenate([[t_do], t_eval[t_eval > t_do]])
    b = simulate(do_love(model, lam), s0[0], s0[1], post)
    if b is None:
        return None, None
    return post, b


# ----------------------------------------------------------------------------
# Классы эквивалентности графов: разные деревья могут задавать одну и ту же функцию
# (например, αx − βxy и −β·x·(y − α/β)). Классом считается набор мономов раскрытой правой части.
# ----------------------------------------------------------------------------
def equiv_key(model: Model, xm=35.0, ym=20.0, rel=0.01):
    """Ключ класса эквивалентности: знаковые мономы каждого уравнения, существенные на данных
    (вклад |c|·E|x^i y^j| ≥ rel от максимального). Для неполиномиальных моделей – структура дерева."""
    parts = []
    try:
        for e in expanded(model, 8):
            t = poly_terms(e)
            if t is None:
                return "nonpoly:" + model.key()
            contrib = {m: abs(c) * xm ** m[0] * ym ** m[1] for m, c in t.items()}
            mx = max(contrib.values()) if contrib else 0
            keep = sorted(m for m, v in contrib.items() if v >= rel * mx and v > 0)
            parts.append(",".join(("+" if t[m] > 0 else "-") + f"x{m[0]}y{m[1]}" for m in keep) or "0")
    except Exception:
        return "err:" + model.key()
    return " | ".join(parts)


def class_representatives(result, ds, acc_max=0.35):
    """Лучший по J граф в каждом классе эквивалентности. Для дешевизны раскрытие считается только для
    моделей с acc < acc_max, остальные – собственный класс. Возвращает (список Scored, {key: число графов в классе})."""
    xm, ym = float(np.mean(ds.x)), float(np.mean(ds.y))
    best, count = {}, {}
    for sc in result.archive.values():
        k = equiv_key(sc.model, xm, ym) if sc.acc < acc_max else "raw:" + sc.key
        count[k] = count.get(k, 0) + 1
        if k not in best or sc.J < best[k].J:
            best[k] = sc
    reps = sorted(best.values(), key=lambda s: s.J)
    for sc in reps:
        k = equiv_key(sc.model, xm, ym) if sc.acc < acc_max else "raw:" + sc.key
        sc.traj = dict(sc.traj); sc.traj["class_key"] = k; sc.traj["class_size"] = count[k]
    return reps, count


def canonical_lv(model: Model):
    """Если раскрытая модель эквивалентна ЛВ – вернуть модель α·x − β·x·y, δ·x·y − γ·y с найденными
    коэффициентами и подписи параметров (α, β, δ, γ), иначе (model, None)."""
    co = lv_coefficients(model)
    if co is None:
        return model.copy(), None
    a, b, d, g = co
    m = true_lv_model_local(a, b, g, d)
    m.meta = dict(model.meta)
    return m, ["α", "β", "δ", "γ"]


def true_lv_model_local(alpha, beta, gamma, delta):
    from .expr import true_lv_model
    return true_lv_model(alpha, beta, gamma, delta)
