import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import zmul_experiment as E

res = json.load(open("zmul_fixed_results.json"))
rows, ens = res["rows"], res["ensemble"]
npz = np.load("zmul_fixed_traj.npz")
base = npz["base"]
AMPS = E.AMPS
names = list(E.SCEN)

# Рис. 1: индикаторы
labels, I, D = [], [], []
for n in names:
    for v in ["x (заяц)", "y (рысь)"]:
        sel = sorted([r for r in rows if r["scenario"] == n and r["var"] == v], key=lambda r: r["a"])
        labels.append(f"{n} · {v.split()[0]}")
        I.append([r["indicator"] for r in sel]); D.append([r["D"] for r in sel])
I, D = np.array(I), np.array(D)
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.imshow(I, cmap=ListedColormap(["#eceff4", "#e4572e"]), vmin=0, vmax=1, aspect="auto")
for i in range(I.shape[0]):
    for j in range(I.shape[1]):
        ax.text(j, i, f"{I[i, j]}\nD={D[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if I[i, j] else "#333")
ax.set_xticks(range(len(AMPS)), [f"{a:g}" for a in AMPS])
ax.set_yticks(range(len(labels)), labels, fontsize=8)
ax.set_xlabel("амплитуда a: z = 1 + a·s(t)  (a=0 — контроль)")
for k in (1.5, 3.5):
    ax.axhline(k, color="white", lw=3)
ax.set_title("Тест КС по траекториям, параметры фиксированы, без мини-батчей\n(1 — p<0.05, Холм, 30 тестов)", fontsize=10)
fig.tight_layout(); fig.savefig("zmul_fig1_indicators.png", dpi=160); plt.close(fig)

# Рис. 2: траектории
cases = [("z_noise", 0.4), ("z_const", 0.4), ("z_sin", 0.4)]
spec = E.SCEN
fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
for r_, (n, a) in enumerate(cases):
    tr = npz[f"{n}|{a}"]
    for c, (obs, nm) in enumerate([(E.HARE, "Заяц x"), (E.LYNX, "Рысь y")]):
        ax = axs[r_, c]
        q = [t for t in rows if t["scenario"] == n and t["a"] == a and t["var"].startswith("xy"[c])][0]
        ax.plot(E.T_GRID, base[c], "k", lw=1.8, label="исходная (z=1)")
        ax.plot(E.T_GRID, tr[c], color="#e4572e", lw=1.8, label=f"с z, a={a:g}")
        ax.scatter(E.T_OBS, obs, s=12, color="#1f77b4", zorder=3, label="данные")
        ax.set_title(f"{n}: {nm}   D={q['D']:.2f}, индикатор={q['indicator']}", fontsize=9); ax.grid(alpha=.3)
    zf = E.make_z(spec[n], a)
    axs[r_, 2].plot(E.T_GRID, [zf(t) for t in E.T_GRID], color="#6a4c93")
    axs[r_, 2].set_title(f"{n}: z(t)", fontsize=9); axs[r_, 2].grid(alpha=.3)
axs[0, 0].legend(fontsize=7)
for ax in axs[-1]:
    ax.set_xlabel("t, лет от 1900")
fig.suptitle("Исходная модель и модель с мультипликативной z (параметры зафиксированы)", fontsize=11)
fig.tight_layout(); fig.savefig("zmul_fig2_trajectories.png", dpi=160); plt.close(fig)

# Рис. 3: ансамбль шума
fig, ax = plt.subplots(figsize=(6.5, 4))
a_ = [e["a"] for e in ens]
ax.plot(a_, [e["share_x"] for e in ens], "o-", label="x (заяц)")
ax.plot(a_, [e["share_y"] for e in ens], "s-", label="y (рысь)")
ax.axhline(0.05, color="gray", ls="--", lw=1)
ax.set_xlabel("амплитуда a"); ax.set_ylabel("доля розыгрышей z с индикатором 1")
ax.set_title("Шумовая z: 50 розыгрышей (траектории, параметры фиксированы)", fontsize=9)
ax.set_ylim(-0.02, 1.02); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("zmul_fig3_noise_ensemble.png", dpi=160); plt.close(fig)
print("ok")
