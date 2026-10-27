import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from experiment_z import Z_SCENARIOS, LAMS, HARE, LYNX, T_OBS, T_GRID

res = json.load(open("results_z.json"))
rows, ens = res["rows"], res["ensemble"]
npz = np.load("results_z.npz")
base = npz["base"]

short = {"z_noise (N(0,1), 1 розыгрыш)": "z_noise (1 розыгрыш)",
         "z_const (z=1)": "z_const (z=1)",
         "z_sin (sin(2πt/10))": "z_sin"}

# --- Рис. 1: индикаторы 0/1 -----------------------------------------------------
names = list(Z_SCENARIOS)
labels, I, D = [], [], []
for n in names:
    for v in ["x (заяц)", "y (рысь)"]:
        sel = [r for r in rows if r["scenario"] == n and r["var"] == v]
        sel.sort(key=lambda r: r["lam"])
        labels.append(f"{short[n]} · {v.split()[0]}")
        I.append([r["indicator"] for r in sel])
        D.append([r["D"] for r in sel])
I, D = np.array(I), np.array(D)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.imshow(I, cmap=ListedColormap(["#eceff4", "#e4572e"]), vmin=0, vmax=1, aspect="auto")
for i in range(I.shape[0]):
    for j in range(I.shape[1]):
        ax.text(j, i, f"{I[i, j]}\nD={D[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if I[i, j] else "#333")
ax.set_xticks(range(len(LAMS)), [f"{l:g}" for l in LAMS])
ax.set_yticks(range(len(labels)), labels, fontsize=8)
ax.set_xlabel("lambda (lam=0 — контроль, совпадает с базовой моделью)")
for k in (1.5, 3.5):
    ax.axhline(k, color="white", lw=3)
ax.set_title("Тест КС по траекториям, параметры фиксированы, без мини-батчей (1 — p<0.05, Холм, 36 тестов)", fontsize=10)
fig.tight_layout()
fig.savefig("fig1_indicators.png", dpi=160)
plt.close(fig)

# --- Рис. 2: траектории базовой модели и модели с z -----------------------------
cases = [("z_noise (N(0,1), 1 розыгрыш)", 0.1), ("z_const (z=1)", 0.05), ("z_sin (sin(2πt/10))", 0.05)]
fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
for r, (n, lam) in enumerate(cases):
    tr = npz[f"{n}|{lam}"]
    ind = {v: [q for q in rows if q["scenario"] == n and q["lam"] == lam and q["var"] == v][0]
           for v in ["x (заяц)", "y (рысь)"]}
    for c, (v, obs, nm) in enumerate([("x (заяц)", HARE, "Заяц x"), ("y (рысь)", LYNX, "Рысь y")]):
        a = axs[r, c]
        a.plot(T_GRID, base[c], "k", lw=1.8, label="исходная (lam=0)")
        a.plot(T_GRID, tr[c], color="#e4572e", lw=1.8, label=f"с z, lam={lam:g}")
        a.scatter(T_OBS, obs, s=12, color="#1f77b4", zorder=3, label="данные")
        q = ind[v]
        a.set_title(f"{short[n]}: {nm}   D={q['D']:.2f}, индикатор={q['indicator']}", fontsize=9)
        a.grid(alpha=.3)
    z = Z_SCENARIOS[n]
    axs[r, 2].plot(T_GRID, [z(t) for t in T_GRID], color="#6a4c93")
    axs[r, 2].set_title(f"{short[n]}: z(t)", fontsize=9)
    axs[r, 2].grid(alpha=.3)
axs[0, 0].legend(fontsize=7)
for a in axs[-1]:
    a.set_xlabel("t, лет от 1900")
fig.suptitle("Исходная модель и модель с переменной z (параметры зафиксированы)", fontsize=11)
fig.tight_layout()
fig.savefig("fig2_trajectories.png", dpi=160)
plt.close(fig)

# --- Рис. 3: доля значимых случаев для шума z (50 розыгрышей) ----------------------
fig, ax = plt.subplots(figsize=(6.5, 4))
l = [e["lam"] for e in ens]
ax.plot(l, [e["share_x"] for e in ens], "o-", label="x (заяц)")
ax.plot(l, [e["share_y"] for e in ens], "s-", label="y (рысь)")
ax.axhline(0.05, color="gray", ls="--", lw=1)
ax.text(l[0], 0.07, "уровень 0.05", color="gray", fontsize=8)
ax.set_xlabel("lambda"); ax.set_ylabel("доля розыгрышей z с индикатором 1")
ax.set_title("Шумовая z: 50 розыгрышей на каждое lambda (траектории, параметры фиксированы)", fontsize=10)
ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3); ax.legend()
fig.tight_layout()
fig.savefig("fig3_noise_ensemble.png", dpi=160)
plt.close(fig)
print("ok")
