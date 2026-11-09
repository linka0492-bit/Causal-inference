"""Do-calculus rules on a time-unrolled causal graph of Lotka-Volterra with z.

Lotka-Volterra has a feedback loop x<->y, so it is not a DAG.
Standard fix: unroll in time. Nodes x_t, y_t, z_t, edges only forward in time.
"""
import networkx as nx
from networkx.algorithms.d_separation import is_d_separator


def unrolled_graph(kind, T=4, z_has_parents=False):
    """kind: 'base' | 'E1' (lam*z inside brackets) | 'E2' (z multiplier) | 'E3' (additive source).
    In E1, E2, E3 z_t is a cause of both x_{t+1} and y_{t+1}.
    z_has_parents=True adds x_t,y_t -> z_t (endogenous z: feedback), used as a positive control for rule 2.
    """
    G = nx.DiGraph()
    for t in range(T):
        G.add_nodes_from([f"x{t}", f"y{t}", f"z{t}"])
    for t in range(T - 1):
        # LV dynamics: each variable depends on both (prey-predator loop)
        for a in "xy":
            for b in "xy":
                G.add_edge(f"{a}{t}", f"{b}{t+1}")
        if kind != "base":
            G.add_edge(f"z{t}", f"x{t+1}")
            G.add_edge(f"z{t}", f"y{t+1}")
    if z_has_parents:
        for t in range(T):
            G.add_edge(f"x{t}", f"z{t}")
            G.add_edge(f"y{t}", f"z{t}")
    return G


def cut_in(G, S):
    H = G.copy()
    for s in S:
        H.remove_edges_from(list(H.in_edges(s)))
    return H


def cut_out(G, S):
    H = G.copy()
    for s in S:
        H.remove_edges_from(list(H.out_edges(s)))
    return H


def rule1(G, Y, Z, X=(), W=()):
    """P(y|do(x),z,w)=P(y|do(x),w) iff Y indep Z | X,W in G_{X-bar}"""
    H = cut_in(G, X)
    return is_d_separator(H, set(Y), set(Z), set(X) | set(W))


def rule2(G, Y, Z, X=(), W=()):
    """P(y|do(x),do(z),w)=P(y|do(x),z,w) iff Y indep Z | X,W in G_{X-bar, Z-underline}"""
    H = cut_out(cut_in(G, X), Z)
    return is_d_separator(H, set(Y), set(Z), set(X) | set(W))


def rule3(G, Y, Z, X=(), W=()):
    """P(y|do(x),do(z),w)=P(y|do(x),w) iff Y indep Z | X,W in G_{X-bar, Z(W)-bar}
    Z(W) = nodes of Z that are not ancestors of W in G_{X-bar}."""
    GX = cut_in(G, X)
    anc_W = set()
    for w in W:
        anc_W |= nx.ancestors(GX, w) | {w}
    ZW = [z for z in Z if z not in anc_W]
    H = cut_in(cut_in(G, X), ZW)
    return is_d_separator(H, set(Y), set(Z), set(X) | set(W))


if __name__ == "__main__":
    for kind in ["base", "E1", "E2", "E3"]:
        G = unrolled_graph(kind)
        Y = ["x3", "y3"]
        print(f"--- {kind}: Y={Y}, Z=['z1'], X=W=[]")
        print("  rule1 (z can be dropped as observation):", rule1(G, Y, ["z1"]))
        print("  rule2 (do(z) == observe z):             ", rule2(G, Y, ["z1"]))
        print("  rule3 (do(z) can be dropped):           ", rule3(G, Y, ["z1"]))
    G = unrolled_graph("E1", z_has_parents=True)
    print("--- E1 with endogenous z (feedback x,y->z)")
    print("  rule2 (do(z) == observe z):", rule2(G, ["x3","y3"], ["z1"]))
    print("  rule3 (do(z) can be dropped):", rule3(G, ["x3","y3"], ["z1"]))
