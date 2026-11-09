import pickle, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docalc import unrolled_graph, rule1, rule2, rule3

r1 = pickle.load(open("rule1.pkl", "rb")); r2 = pickle.load(open("rule2_N4000.pkl", "rb")); r3 = pickle.load(open("rule3.pkl", "rb"))
Y, Z = ["x2", "y2"], ["z1"]
def oracle(kind, endo):
    G = unrolled_graph(kind, T=4, z_has_parents=endo)
    return dict(R1=rule1(G, Y, Z, W=["x1", "y1"]), R2=rule2(G, Y, Z), R3=rule3(G, Y, Z))

rows = []
# rule 1 and rule 2: z endogenous or exogenous
for a, b in zip(r1, r2):
    kind = "base" if a["exp"] == "ctrl" else a["exp"]; endo = a["rho"] > 0
    o = oracle(kind, endo)
    rows.append(dict(rule="R1", exp=a["exp"], rho=a["rho"], kind="noise", s=a["s"], graph_holds=o["R1"], data_rejects=bool(a["F_sig"]),
                     stat=f"F p={a['F']:.1e}; dR2={a['dR2']:.3f}"))
    rows.append(dict(rule="R2", exp=b["exp"], rho=b["rho"], kind="noise", s=b["s"], graph_holds=o["R2"], data_rejects=b["n_sig"] > 0,
                     stat=f"KS sig {b['n_sig']}/{b['n_tests']}; maxD={b['maxD']:.2f}"))
for r in r3:
    if r["exp"] == "ctrl": continue
    o = oracle(r["exp"], False)
    rows.append(dict(rule="R3", exp=r["exp"], rho=0.0, kind=r["kind"], s=r["s"], graph_holds=o["R3"], data_rejects=r["ks_nsig"] > 0,
                     stat=f"KS sig {r['ks_nsig']}/40; maxD={r['ks_maxD']:.2f}; pooled D={r['pooled_D']:.2f}"))
for x in rows:
    x["agree"] = (x["graph_holds"] != x["data_rejects"])      # rule holds <=> data does NOT reject
with open("agreement.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for R in ["R1", "R2", "R3"]:
    sub = [x for x in rows if x["rule"] == R]; ag = sum(x["agree"] for x in sub)
    print(R, "согласие граф/данные: %d из %d" % (ag, len(sub)))
    for x in sub:
        if not x["agree"]: print("   расхождение:", x["exp"], "rho=%s" % x["rho"], x["kind"], x["s"], "| граф: правило", "выполняется" if x["graph_holds"] else "нарушается", "| данные:", "отвергают" if x["data_rejects"] else "не отвергают", "|", x["stat"])

# figure
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
def heat(a, data, rows_l, cols_l, title, fmt):
    a.imshow(np.array(data, float), cmap="Reds", vmin=0, vmax=1, aspect="auto")
    a.set_xticks(range(len(cols_l))); a.set_xticklabels(cols_l); a.set_yticks(range(len(rows_l))); a.set_yticklabels(rows_l, fontsize=8)
    for i, row in enumerate(fmt):
        for j, t in enumerate(row): a.text(j, i, t, ha="center", va="center", fontsize=7, color=("white" if np.array(data, float)[i][j] > 0.45 else "black"))
    a.set_title(title, fontsize=10)
# R3: rows exp/kind, cols 4 strength levels
lab, dat, fm = [], [], []
for exp in ["E1", "E2", "E3"]:
    for kind in ["noise", "const", "sin"]:
        sub = [r for r in r3 if r["exp"] == exp and r["kind"] == kind]
        lab.append(f"{exp} {kind}"); dat.append([r["ks_nsig"]/40 for r in sub]); fm.append([f"{r['ks_nsig']}/40\nD={r['ks_maxD']:.2f}" for r in sub])
heat(ax[0], dat, lab, ["сила 1", "сила 2", "сила 3", "сила 4"], "Правило 3: КС по годам (значимых лет из 40)", fm)
lab, dat, fm = [], [], []
for exp in ["E1", "E2", "E3"]:
    for rho in [0.0, 0.8]:
        sub = [r for r in r1 if r["exp"] == exp and r["rho"] == rho]
        lab.append(f"{exp} rho={rho}"); dat.append([float(r["F_sig"]) for r in sub]); fm.append([f"R²+{r['dR2']*100:.1f}%" for r in sub])
heat(ax[1], dat, lab, ["сила 1", "сила 2", "сила 3", "сила 4"], "Правило 1: F-тест (1 = z нельзя убрать)", fm)
lab, dat, fm = [], [], []
for exp in ["E1", "E2", "E3"]:
    for rho in [0.0, 0.8]:
        sub = [r for r in r2 if r["exp"] == exp and r["rho"] == rho]
        lab.append(f"{exp} rho={rho}"); dat.append([r["n_sig"]/r["n_tests"] for r in sub]); fm.append([f"{r['n_sig']}/{r['n_tests']}\nD={r['maxD']:.2f}" for r in sub])
heat(ax[2], dat, lab, ["сила 1", "сила 2", "сила 3", "сила 4"], "Правило 2: КС do(z)=c и наблюдение z=c (N=4000)", fm)
plt.tight_layout(); plt.savefig("three_rules.png", dpi=130)
