import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import ks_2samp

import minibatch_z as mb

res = json.load(open("minibatch_results.json"))
rows, basest = res["rows"], res["base"]
S = np.load("minibatch_samples.npz")
SCEN = ["z_noise", "z_const", "z_sin"]
LAMS = [0.01, 0.02, 0.05, 0.1]
P = mb.P_NAMES
GL = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "x0": "x₀", "y0": "y₀"}

# --- Рис. 4: индикаторы по параметрам ------------------------------------------
cfg = [("control", 0.0)] + [(s, l) for s in SCEN for l in LAMS]
I = np.zeros((len(cfg), 6), int); D = np.zeros((len(cfg), 6)); W = np.zeros((len(cfg), 6))
for i, (s, l) in enumerate(cfg):
    for j, p in enumerate(P):
        r = [q for q in rows if q["scenario"] == s and q["lam"] == l and q["param"] == p][0]
        I[i, j], D[i, j], W[i, j] = r["indicator"], r["D"], r["w1_sigma"]
fig, ax = plt.subplots(figsize=(9, 7))
ax.imshow(I, cmap=ListedColormap(["#eceff4", "#e4572e"]), vmin=0, vmax=1, aspect="auto")
for i in range(len(cfg)):
    for j in range(6):
        ax.text(j, i, f"{I[i, j]}\nD={D[i, j]:.2f}  W/σ={W[i, j]:.1f}", ha="center", va="center",
                fontsize=7, color="white" if I[i, j] else "#333")
ax.set_xticks(range(6), [GL[p] for p in P], fontsize=11)
ax.set_yticks(range(len(cfg)), ["контроль (base vs base)" if s == "control" else f"{s}, λ={l:g}"
                                for s, l in cfg], fontsize=8)
for k in (0.5, 4.5, 8.5):
    ax.axhline(k, color="white", lw=3)
ax.set_title(f"Мини-батчи (100 батчей по 12 из 21): индикатор КС по параметрам\n"
             f"1 — p<0.05 с поправкой Холма ({res['n_tests']} тестов); D и W1/σ(base) в ячейках", fontsize=10)
fig.tight_layout(); fig.savefig("fig4_param_indicators.png", dpi=160); plt.close(fig)

# --- Рис. 5: сдвиг оценок параметров (среднее ± σ по батчам) -------------------
fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
col = {"z_noise": "#6a4c93", "z_const": "#1f77b4", "z_sin": "#e4572e"}
for a, (j, p) in zip(axs.ravel(), enumerate(P)):
    b = basest[p]
    a.axhspan(b["mean"] - b["std"], b["mean"] + b["std"], color="gray", alpha=.18, label="base ±σ")
    a.axhline(b["mean"], color="k", ls="--", lw=1)
    for s in SCEN:
        m, sd, st = [], [], []
        for l in LAMS:
            r = [q for q in rows if q["scenario"] == s and q["lam"] == l and q["param"] == p][0]
            m.append(r["mean"]); sd.append(r["std"]); st.append(r["indicator"])
        a.errorbar(LAMS, m, yerr=sd, fmt="o-", color=col[s], capsize=2, lw=1.3, ms=4, label=s)
        for l, mm, ss in zip(LAMS, m, st):
            if ss:
                a.plot(l, mm, "*", color="red", ms=12, zorder=5, mfc="none", mew=1.5)
    a.set_title(GL[p]); a.set_xlabel("λ"); a.grid(alpha=.3)
axs[0, 0].legend(fontsize=7)
fig.suptitle("Оценки параметров по батчам при разных λ (красная звезда — значимо отличается от base по КС)",
             fontsize=10)
fig.tight_layout(); fig.savefig("fig5_param_shift.png", dpi=160); plt.close(fig)

# --- Рис. 6: траектории при усреднённых по батчам параметрах -------------------
TG = np.arange(0, 20.0001, 0.1)
bm = S["base|0.0"].mean(axis=0)
base_tr = mb.simulate(bm, 0.0, mb.make_z(("none",)), TG)
specs = {"z_noise": ("noise", 0), "z_const": ("const",), "z_sin": ("sin",)}
cases = [("z_noise", 0.1), ("z_const", 0.05), ("z_sin", 0.05)]
# Тест КС по траекториям с параметрами, усреднёнными по батчам (24 теста, Холм)
tt = []
for s in SCEN:
    for l in LAMS:
        tr = mb.simulate(S[f"{s}|{l}"].mean(axis=0), l, mb.make_z(specs[s]), TG)
        for j in range(2):
            r = ks_2samp(base_tr[j], tr[j])
            tt.append(dict(scenario=s, lam=l, var=["x", "y"][j], D=float(r.statistic), p=float(r.pvalue)))
sg = mb.holm([t["p"] for t in tt])
for t, s_ in zip(tt, sg):
    t["indicator"] = int(s_)
json.dump(tt, open("minibatch_traj_ks.json", "w"), ensure_ascii=False, indent=1)

fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
for r_, (s, l) in enumerate(cases):
    zf = mb.make_z(specs[s])
    tr = mb.simulate(S[f"{s}|{l}"].mean(axis=0), l, zf, TG)
    for c, (obs, nm) in enumerate([(mb.HARE, "Заяц x"), (mb.LYNX, "Рысь y")]):
        a = axs[r_, c]
        q = [t for t in tt if t["scenario"] == s and t["lam"] == l and t["var"] == "xy"[c]][0]
        a.plot(TG, base_tr[c], "k", lw=1.8, label="исходная (lam=0), средние параметры")
        a.plot(TG, tr[c], color="#e4572e", lw=1.8, label=f"с z, λ={l:g}, переподогнанные параметры")
        a.scatter(mb.T_OBS, obs, s=12, color="#1f77b4", zorder=3, label="данные")
        a.set_title(f"{s}: {nm}   D={q['D']:.2f}, индикатор={q['indicator']}", fontsize=9)
        a.grid(alpha=.3)
    axs[r_, 2].plot(TG, [zf(t) for t in TG], color="#6a4c93")
    axs[r_, 2].set_title(f"{s}: z(t)", fontsize=9); axs[r_, 2].grid(alpha=.3)
axs[0, 0].legend(fontsize=6, loc="upper right")
for a in axs[-1]:
    a.set_xlabel("t, лет от 1900")
fig.suptitle("Исходная модель и модель с z после переподгонки параметров (среднее по 100 батчам)", fontsize=11)
fig.tight_layout(); fig.savefig("fig6_trajectories_refit.png", dpi=160); plt.close(fig)

# --- Рис. 7: сравнение двух тестов (фиксированные параметры vs переподгонка) ----
fixed = json.load(open("results_z.json"))["rows"]
nm_fixed = {"z_noise": "z_noise (N(0,1), 1 розыгрыш)", "z_const": "z_const (z=1)", "z_sin": "z_sin (sin(2πt/10))"}
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for a, s in zip(axs, SCEN):
    f = [np.mean([r["D"] for r in fixed if r["scenario"] == nm_fixed[s] and r["lam"] == l]) for l in LAMS]
    g = [np.mean([t["D"] for t in tt if t["scenario"] == s and t["lam"] == l]) for l in LAMS]
    h = [np.mean([r["D"] for r in rows if r["scenario"] == s and r["lam"] == l]) for l in LAMS]
    a.plot(LAMS, f, "o-", label="траектории, параметры фиксированы")
    a.plot(LAMS, g, "s-", label="траектории после переподгонки")
    a.plot(LAMS, h, "^-", label="параметры по батчам (средн. D по 6)")
    a.set_title(s); a.set_xlabel("λ"); a.grid(alpha=.3)
axs[0].set_ylabel("статистика D"); axs[0].legend(fontsize=7)
fig.suptitle("Статистика D: траектории (параметры фиксированы / после переподгонки) и распределение параметров по батчам", fontsize=10)
fig.tight_layout(); fig.savefig("fig7_compare_tests.png", dpi=160); plt.close(fig)
print("ok")
for t in tt:
    if t["lam"] in (0.05, 0.1):
        print(t)
