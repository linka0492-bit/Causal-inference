"""
Параметры как многомерная случайная величина: распределение оценок по мини-батчам.

Две процедуры (обе на тех же данных Hudson Bay и той же модели из дерева):

  A) «Оценка на каждом мини-батче». Функция make_minibatches делит выборку на
     случайные подвыборки наблюдаемых моментов времени. На каждом батче подгоняем
     вектор theta = (alpha, beta, gamma, delta, x0, y0) (невязки -- только по точкам
     батча). Получаем выборку theta_1..theta_B, из неё: мат. ожидание, ковариация,
     гистограммы по каждой компоненте.

  B) Настоящий стохастический градиентный спуск (Adam) по мини-батчам: на каждом шаге
     берётся новый батч, градиент считается только по нему. Итерации после «прогрева»
     -- это тоже выборка из (стационарного) распределения параметров; по ней строятся
     гистограммы. Оптимизация идёт в log-координатах (положительность параметров).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lv_common import LVModel, fit, T_OBS, DATA, PARAM_NAMES, P_START

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

LABELS = [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$", r"$x_0$", r"$y_0$"]


def make_minibatches(n_points, batch_size, n_batches, rng):
    """Делит выборку на мини-батчи: каждый -- случайное подмножество индексов
    наблюдений без повторов внутри батча (между батчами повторы допустимы)."""
    return [np.sort(rng.choice(n_points, size=batch_size, replace=False)) for _ in range(n_batches)]


def minibatch_estimates(model, p_full, batches):
    est, rss_b = [], []
    for k, idx in enumerate(batches):
        p, rss, res = fit(model, p0=p_full, idx=idx, method="trf", max_nfev=100)
        est.append(p)
        rss_b.append(rss / len(idx))
    return np.array(est), np.array(rss_b)


def residual_vec(model, theta):
    return model.simulate(theta, T_OBS) - DATA  # (21, 2)


def sgd_adam(model, p_start, batch_size, n_steps, lr, rng, beta1=0.9, beta2=0.999, eps=1e-8):
    """SGD (Adam) в log-координатах phi = log theta; градиент только по мини-батчу."""
    phi = np.log(p_start)
    m = np.zeros_like(phi)
    v = np.zeros_like(phi)
    h = 1e-5
    traj = np.empty((n_steps, len(phi)))
    for k in range(n_steps):
        theta = np.exp(phi)
        r0 = residual_vec(model, theta)
        J = np.empty((r0.size, len(phi)))  # d r / d phi, прямые разности
        for j in range(len(phi)):
            ph = phi.copy()
            ph[j] += h
            J[:, j] = ((residual_vec(model, np.exp(ph)) - r0) / h).ravel()
        idx = rng.choice(len(T_OBS), size=batch_size, replace=False)
        rows = np.concatenate([2 * idx, 2 * idx + 1])  # ravel() по (21,2): [h0,l0,h1,l1,...]
        g = 2 * J[rows].T @ r0.ravel()[rows] / len(idx)  # градиент среднего по батчу RSS
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh, vh = m / (1 - beta1 ** (k + 1)), v / (1 - beta2 ** (k + 1))
        phi = phi - lr * mh / (np.sqrt(vh) + eps)
        traj[k] = np.exp(phi)
    return traj


def hist_figure(samples, ref, title, path, color="tab:blue"):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, ax in enumerate(axes.ravel()):
        s = samples[:, j]
        ax.hist(s, bins=30, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(s.mean(), color="black", lw=2, label=f"среднее по батчам = {s.mean():.3g}")
        ax.axvline(ref[j], color="tab:red", ls="--", lw=2, label=f"полная выборка = {ref[j]:.3g}")
        ax.set_title(f"{LABELS[j]}   (σ = {s.std(ddof=1):.2g})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def describe(samples, ref, name):
    print(f"\n{name}: n={len(samples)}")
    print("  параметр |  E[theta]  |   std   | 2.5% .. 97.5%        | полная выборка")
    for j, pn in enumerate(PARAM_NAMES):
        s = samples[:, j]
        lo, hi = np.percentile(s, [2.5, 97.5])
        print(f"  {pn:8s} | {s.mean():10.4f} | {s.std(ddof=1):7.4f} | {lo:8.4f} .. {hi:8.4f} | {ref[j]:10.4f}")
    corr = np.corrcoef(samples.T)
    print("  корреляции |", " ".join(f"{n:>6s}" for n in PARAM_NAMES))
    for j, pn in enumerate(PARAM_NAMES):
        print(f"  {pn:10s} |", " ".join(f"{corr[j, k]:6.2f}" for k in range(6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--n-batches", type=int, default=300)
    ap.add_argument("--sgd-steps", type=int, default=1500)
    ap.add_argument("--sgd-burn", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-sgd", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    model = LVModel(rtol=1e-7, atol=1e-7)
    p_full, rss_full, _ = fit(model)
    print("Полная выборка (LM):", np.round(p_full, 4), f"RSS={rss_full:.1f}")

    # ------------------------------ A ------------------------------
    batches = make_minibatches(len(T_OBS), a.batch_size, a.n_batches, rng)
    est, _ = minibatch_estimates(model, p_full, batches)
    np.savetxt(OUT / "minibatch_estimates.csv", est, delimiter=",", header=",".join(PARAM_NAMES), comments="")
    describe(est, p_full, f"A) оценка на каждом мини-батче (размер {a.batch_size} из {len(T_OBS)})")
    hist_figure(est, p_full,
                f"Распределение параметров по {a.n_batches} мини-батчам (размер {a.batch_size} из {len(T_OBS)} точек)",
                OUT / "param_hist_minibatch_fits.png")

    # ------------------------------ B ------------------------------
    if not a.skip_sgd:
        print("\nB) SGD (Adam) по мини-батчам...")
        traj = sgd_adam(model, p_full, a.batch_size, a.sgd_steps, a.lr, rng)
        np.savetxt(OUT / "sgd_trajectory.csv", traj, delimiter=",", header=",".join(PARAM_NAMES), comments="")
        post = traj[a.sgd_burn:]
        describe(post, p_full, f"B) итерации SGD после прогрева ({a.sgd_burn} шагов), lr={a.lr}")
        hist_figure(post, p_full,
                    f"Распределение итераций SGD (Adam, lr={a.lr}, батч {a.batch_size}), шаги {a.sgd_burn}..{a.sgd_steps}",
                    OUT / "param_hist_sgd.png", color="tab:green")

        fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
        for j, ax in enumerate(axes.ravel()):
            ax.plot(traj[:, j], lw=0.8)
            ax.axhline(p_full[j], color="tab:red", ls="--")
            ax.axvline(a.sgd_burn, color="gray", ls=":")
            ax.set_title(LABELS[j])
            ax.grid(alpha=0.3)
        fig.suptitle("Траектории параметров в SGD (красная -- LM по полной выборке)")
        plt.tight_layout()
        plt.savefig(OUT / "sgd_trace.png", dpi=130)
        plt.close(fig)

    print(f"\nГотово, графики и csv в {OUT}")


if __name__ == "__main__":
    main()
