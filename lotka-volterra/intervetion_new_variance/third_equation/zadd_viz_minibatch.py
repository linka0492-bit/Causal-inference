import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import ks_2samp

import zadd_experiment as E

res = json.load(open("zadd_minibatch_results.json"))
rows, basest = res["rows"], res["base"]
S = np.load("zadd_minibatch_samples.npz")
SC = list(E.SCEN)
AM = E.AMPS[1:]
P = E.P_NAMES
GL = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "x0": "x₀", "y0": "y₀"}

# Рис. 4: индикаторы по параметрам
cfg = [("control", 0.0)] + [(s, a) for s in SC for a in AM]
I = np.zeros((len(cfg), 6), int); D = np.zeros((len(cfg), 6)); W = np.zeros((len(cfg), 6))
for i, (s, a) in enumerate(cfg):
    for j, p in enumerate(P):
        r = [q for q in rows if q["scenario"] == s and q["e"] == a and q["param"] == p][0]
        I[i, j], D[i, j], W[i, j] = r["indicator"], r["D"], r["w1_sigma"]
fig, ax = plt.subplots(figsize=(9, 7))
ax.imshow(I, cmap=ListedColormap(["#eceff4", "#e4572e"]), vmin=0, vmax=1, aspect="auto")
for i in range(len(cfg)):
    for j in range(6):
        ax.text(j, i, f"{I[i, j]}\nD={D[i, j]:.2f}  W/σ={W[i, j]:.1f}", ha="center", va="center",
                fontsize=7, color="white" if I[i, j] else "#333")
ax.set_xticks(range(6), [GL[p] for p in P], fontsize=11)
ax.set_yticks(range(len(cfg)), ["контроль (base vs base)" if s == "control" else f"{s}, e={a:g}"
                                for s, a in cfg], fontsize=8)
for k in (0.5, 4.5, 8.5):
    ax.axhline(k, color="white", lw=3)
ax.set_title(f"Мини-батчи (100 батчей по 12 из 21): индикатор КС по параметрам\n"
             f"1 — p<0.05 с поправкой Холма ({res['n_tests']} тестов); D и W1/σ(base) в ячейках", fontsize=10)
fig.tight_layout(); fig.savefig("zadd_fig4_param_indicators.png", dpi=160); plt.close(fig)

# Рис. 5: сдвиг оценок (медиана, 16–84 перцентили — устойчиво к редким неудачным подгонкам)
fig, axs = plt.subplots(2, 3, figsize=(12, 6.5))
col = {"z_noise": "#6a4c93", "z_const": "#1f77b4", "z_sin": "#e4572e"}
B = S["base|0.0"]
for ax, (j, p) in zip(axs.ravel(), enumerate(P)):
    q16, q50, q84 = np.percentile(B[:, j], [16, 50, 84])
    ax.axhspan(q16, q84, color="gray", alpha=.18, label="base 16–84%")
    ax.axhline(q50, color="k", ls="--", lw=1)
    for s_ in SC:
        v = [np.percentile(S[f"{s_}|{a}"][:, j], [16, 50, 84]) for a in AM]
        v = np.array(v)
        ax.errorbar(AM, v[:, 1], yerr=[v[:, 1] - v[:, 0], v[:, 2] - v[:, 1]], fmt="o-", color=col[s_],
                    capsize=2, lw=1.3, ms=4, label=s_)
        for a, mm in zip(AM, v[:, 1]):
            r = [q for q in rows if q["scenario"] == s_ and q["e"] == a and q["param"] == p][0]
            if r["indicator"]:
                ax.plot(a, mm, "*", color="red", ms=12, zorder=5, mfc="none", mew=1.5)
    ax.set_title(GL[p]); ax.set_xlabel("сила вмешательства e"); ax.grid(alpha=.3)
axs[0, 0].legend(fontsize=7)
fig.suptitle("Оценки параметров по батчам при разных e: медиана и 16–84% (красная звезда — значимо отличается от base по КС)", fontsize=10)
fig.tight_layout(); fig.savefig("zadd_fig5_param_shift.png", dpi=160); plt.close(fig)

# Рис. 6: траектории после переподгонки (средние по батчам параметры)
TG = E.T_GRID
bm = S["base|0.0"].mean(axis=0)
base_tr = E.simulate(bm, E.make_z(("none",), 0), TG)
tt = []
for s in SC:
    for a in AM:
        tr = E.simulate(S[f"{s}|{a}"].mean(axis=0), E.make_z(E.SCEN[s], a), TG)
        for j in range(2):
            r = ks_2samp(base_tr[j], tr[j])
            tt.append(dict(scenario=s, e=a, var="xy"[j], D=float(r.statistic), p=float(r.pvalue)))
for t, s_ in zip(tt, E.holm([t["p"] for t in tt])):
    t["indicator"] = int(s_)
json.dump(tt, open("zadd_minibatch_traj_ks.json", "w"), ensure_ascii=False, indent=1)
cases = [("z_noise", 1.0), ("z_const", 1.0), ("z_sin", 1.0)]
fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
for r_, (s, a) in enumerate(cases):
    zf = E.make_z(E.SCEN[s], a)
    tr = E.simulate(S[f"{s}|{a}"].mean(axis=0), zf, TG)
    for c, (obs, nm) in enumerate([(E.HARE, "Заяц x"), (E.LYNX, "Рысь y")]):
        ax = axs[r_, c]
        q = [t for t in tt if t["scenario"] == s and t["e"] == a and t["var"] == "xy"[c]][0]
        ax.plot(TG, base_tr[c], "k", lw=1.8, label="исходная (e=0), средние параметры")
        ax.plot(TG, tr[c], color="#e4572e", lw=1.8, label=f"с z, e={a:g}, переподогнанные параметры")
        ax.scatter(E.T_OBS, obs, s=12, color="#1f77b4", zorder=3, label="данные")
        ax.set_title(f"{s}: {nm}   D={q['D']:.2f}, индикатор={q['indicator']}", fontsize=9); ax.grid(alpha=.3)
    axs[r_, 2].plot(TG, [zf(t) for t in TG], color="#6a4c93")
    axs[r_, 2].set_title(f"{s}: z(t)", fontsize=9); axs[r_, 2].grid(alpha=.3)
axs[0, 0].legend(fontsize=6, loc="upper right")
for ax in axs[-1]:
    ax.set_xlabel("t, лет от 1900")
fig.suptitle("Исходная модель и модель с z после переподгонки параметров (среднее по 100 батчам)", fontsize=11)
fig.tight_layout(); fig.savefig("zadd_fig6_trajectories_refit.png", dpi=160); plt.close(fig)

# Рис. 7: сравнение D трёх тестов
fixed = json.load(open("zadd_fixed_results.json"))["rows"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, s in zip(axs, SC):
    f = [np.mean([r["D"] for r in fixed if r["scenario"] == s and r["e"] == a]) for a in AM]
    g = [np.mean([t["D"] for t in tt if t["scenario"] == s and t["e"] == a]) for a in AM]
    h = [np.mean([r["D"] for r in rows if r["scenario"] == s and r["e"] == a]) for a in AM]
    ax.plot(AM, f, "o-", label="траектории, параметры фиксированы")
    ax.plot(AM, g, "s-", label="траектории после переподгонки")
    ax.plot(AM, h, "^-", label="параметры по батчам (средн. D по 6)")
    ax.set_title(s); ax.set_xlabel("сила вмешательства e"); ax.grid(alpha=.3)
axs[0].set_ylabel("статистика D"); axs[0].legend(fontsize=7)
fig.suptitle("Статистика D: траектории (параметры фиксированы / после переподгонки) и распределение параметров по батчам", fontsize=10)
fig.tight_layout(); fig.savefig("zadd_fig7_compare_tests.png", dpi=160); plt.close(fig)
print("ok")
for t in tt:
    if t["e"] in (0.5, 1.0):
        print(t)
