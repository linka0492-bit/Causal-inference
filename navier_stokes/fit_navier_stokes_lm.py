"""
Шаг 3 эксперимента для 2D несжимаемых уравнений Навье-Стокса:
настройка параметра вязкости nu (и, опционально, амплитуды начального
возмущения) методом Левенберга-Марквардта на реальной траектории из
датасета The Well (shear_flow).

Схема полностью повторяет fit_lotka_volterra_lm.py:
  1. загрузка реальной выборки (здесь — .npz-срез The Well);
  2. компиляция «дерева» (здесь — псевдоспектральный решатель
     NavierStokes2D из navier_stokes_solver.py) в численную правую часть,
     параметризованную вектором theta;
  3. функция невязки residuals для scipy.optimize.least_squares;
  4. least_squares(method="lm") — классический Левенберг-Марквардт
     (MINPACK LMDIF под капотом);
  5. вывод подобранных параметров и финальное сравнение данные-vs-модель.

Невязка берётся по интегральной диагностике — кинетической энергии
E(t) = 0.5 * mean(vx^2 + vy^2) на каждом сохранённом шаге. Это
скалярный временной ряд, что делает задачу наименьших квадратов
хорошо обусловленной и устойчивой (полная невязка по полям имела бы
размерность ~ n^2 * n_steps и требовала бы огромного числа вызовов
решателя).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from navier_stokes_solver import NavierStokes2D, make_shear_layer_ic


# ---------------------------------------------------------------------------
# 1. Загрузка реальной траектории (срез The Well "shear_flow")
# ---------------------------------------------------------------------------
DATA_PATH = "well_shear_flow_slice.npz"


def load_well_slice(path=DATA_PATH):
    """
    Загружает срез датасета The Well и приводит его к единому виду.

    Ожидаемые ключи (могут отличаться в зависимости от версии The Well):
      - 'vx' : (n_steps, n, n)  — x-компонента скорости
      - 'vy' : (n_steps, n, n)  — y-компонента скорости
      - 't'  : (n_steps,)       — времена сохранённых кадров (если есть)
      - 'nu' : скаляр            — истинная вязкость (если есть)

    Возвращает dict с полями vx, vy, t, n, L, nu_true.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл {path} не найден. Сначала запустите prepare_well_slice.py "
            "на локальной машине с доступом в интернет, чтобы скачать срез "
            "The Well, или положите .npz-файл рядом со скриптом."
        )

    raw = np.load(path)
    keys = list(raw.keys())
    print(f"Ключи в {path}: {keys}")

    # --- извлекаем поля скорости ------------------------------------------
    if "vx" in raw and "vy" in raw:
        vx = raw["vx"]
        vy = raw["vy"]
    elif "velocity" in raw:
        # The Well хранит скорость как (n_steps, n, n, 2)
        vel = raw["velocity"]
        vx, vy = vel[..., 0], vel[..., 1]
    elif "input_fields" in raw and "output_fields" in raw:
        # В актуальной версии The Well хранит одно временное сечение как
        # (1, H, W, 4), где первые две компоненты представляют (vx, vy),
        # а выходное поле даёт следующее состояние на временном шаге.
        inp = np.asarray(raw["input_fields"])
        out = np.asarray(raw["output_fields"])
        if inp.shape[-1] < 2:
            raise KeyError(
                "Сигнатура input_fields/output_fields не содержит компоненты скорости. "
                "Проверьте структуру .npz-файла."
            )
        vx = np.stack([inp[0, ..., 0], out[0, ..., 0]], axis=0)
        vy = np.stack([inp[0, ..., 1], out[0, ..., 1]], axis=0)
    else:
        raise KeyError(
            "Не найдены ключи 'vx'/'vy', 'velocity' или 'input_fields'/'output_fields'. "
            "Проверьте структуру .npz-файла."
        )

    # --- времена -----------------------------------------------------------
    if "t" in raw:
        t = np.asarray(raw["t"], dtype=float)
    elif "input_time_grid" in raw and "output_time_grid" in raw:
        t = np.asarray([raw["input_time_grid"][0], raw["output_time_grid"][0]], dtype=float)
    else:
        # если времён нет, используем равномерную сетку (заглушка)
        n_steps = vx.shape[0]
        t = np.arange(n_steps, dtype=float)
        print("Предупреждение: ключ 't' отсутствует, используется "
              "равномерная сетка по индексам кадров.")

    # --- истинная вязкость (если есть) -------------------------------------
    nu_true = None
    if "nu" in raw:
        nu_true = float(raw["nu"])
    elif "constant_scalars" in raw:
        cs = np.asarray(raw["constant_scalars"])
        if cs.size >= 2 and np.isfinite(cs[1]):
            nu_true = float(cs[1])
    if nu_true is not None:
        print(f"Истинная вязкость в данных: nu = {nu_true:.6f}")

    if vx.ndim >= 3:
        n = min(vx.shape[1], vx.shape[2])
        vx = vx[:, :n, :n]
        vy = vy[:, :n, :n]
    else:
        n = vx.shape[1]
    L = 2 * np.pi  # периодическая область, как в решателе
    print(f"Загружена траектория: vx.shape = {vx.shape}, n = {n}, "
          f"n_steps = {vx.shape[0]}")

    return dict(vx=vx, vy=vy, t=t, n=n, L=L, nu_true=nu_true)


# ---------------------------------------------------------------------------
# 2. Обёртка над решателем: симуляция для заданного theta
# ---------------------------------------------------------------------------
def simulate_ns(theta, vx0, vy0, t_eval, n, L, dt_sim):
    """
    Запускает псевдоспектральный решатель с вязкостью nu = theta[0]
    (при необходимости theta может содержать дополнительные параметры,
    например амплитуду начального возмущения).

    Возвращает массив кинетической энергии E(t) той же длины, что t_eval.
    """
    nu = theta[0]

    solver = NavierStokes2D(n=n, L=L)

    # число шагов интегрирования между сохранёнными кадрами
    if len(t_eval) > 1:
        dt_data = t_eval[1] - t_eval[0]
    else:
        dt_data = 1.0
    n_sub = max(1, int(round(dt_data / dt_sim)))

    # полное число шагов и частота сохранения
    n_steps_total = (len(t_eval) - 1) * n_sub
    save_every = n_sub

    traj_x, traj_y = solver.simulate(
        vx0, vy0, nu=nu, dt=dt_sim,
        n_steps=n_steps_total, save_every=save_every,
    )

    # диагностика: кинетическая энергия на каждом сохранённом кадре
    energies = np.array([
        solver.kinetic_energy(traj_x[i], traj_y[i])
        for i in range(traj_x.shape[0])
    ])

    # приводим длину к len(t_eval): обрезаем или дополняем последним значением
    if len(energies) > len(t_eval):
        energies = energies[:len(t_eval)]
    elif len(energies) < len(t_eval):
        pad = np.full(len(t_eval) - len(energies), energies[-1])
        energies = np.concatenate([energies, pad])

    return energies


# ---------------------------------------------------------------------------
# 3. Функция невязки для least_squares
# ---------------------------------------------------------------------------
def residuals(params_vec, vx0, vy0, t_eval, data_energy, n, L, dt_sim):
    """
    params_vec = [nu]  (при необходимости можно расширить до [nu, amplitude])

    Невязка — разность кинетических энергий:
        r_i = E_model(t_i; nu) - E_data(t_i)

    Такая скалярная невязка даёт хорошо обусловленную задачу
    наименьших квадратов и не требует хранения полных полей.
    """
    nu = params_vec[0]

    model_energy = simulate_ns(
        [nu], vx0, vy0, t_eval, n, L, dt_sim
    )

    return model_energy - data_energy


# ---------------------------------------------------------------------------
# 4. Основной сценарий
# ---------------------------------------------------------------------------
def main():
    # --- 4.1. загрузка реальной траектории ---------------------------------
    data = load_well_slice()
    vx_data = data["vx"]          # (n_steps, n, n)
    vy_data = data["vy"]
    t_data = data["t"]
    n = data["n"]
    L = data["L"]
    nu_true = data["nu_true"]

    # диагностика данных: кинетическая энергия на каждом кадре
    data_energy = 0.5 * np.mean(vx_data ** 2 + vy_data ** 2, axis=(1, 2))

    # --- 4.2. начальные условия: первый кадр реальной траектории ----------
    vx0 = vx_data[0].copy()
    vy0 = vy_data[0].copy()

    # шаг интегрирования решателя (можно уменьшить для точности)
    dt_sim = 0.01
    # при необходимости пересэмплируем t_data под шаг dt_sim
    if len(t_data) > 1:
        dt_data = t_data[1] - t_data[0]
        if abs(dt_data - dt_sim) > 1e-12:
            print(f"Шаг данных dt={dt_data:.4f} отличается от dt_sim={dt_sim}. "
                  f"Решатель будет использовать внутренние подшаги.")
    else:
        dt_data = dt_sim
        t_data = np.array([0.0, dt_sim])
        data_energy = np.array([data_energy[0], data_energy[0]])

    # --- 4.3. стартовая точка для LM ---------------------------------------
    # если в данных есть nu — стартуем от него, иначе от разумного значения
    nu0 = nu_true if nu_true is not None else 0.01
    p0 = np.array([nu0])

    print("\nСтартовая точка (из данных или по умолчанию):")
    print(f"  nu = {p0[0]:.6f}")

    # --- 4.4. оптимизация Левенберга-Марквардта ---------------------------
    result = least_squares(
        residuals,
        p0,
        method="lm",
        max_nfev=200,
        args=(vx0, vy0, t_data, data_energy, n, L, dt_sim),
    )

    nu_fit = result.x[0]

    print("\nПодобранные параметры (Левенберг-Марквардт, scipy MINPACK LMDIF):")
    print(f"  nu = {nu_fit:.6f}")
    if nu_true is not None:
        print(f"  (истинная вязкость в данных: nu = {nu_true:.6f})")

    print(f"\nУспех оптимизации: {result.success}, сообщение: {result.message}")
    print(f"Число вызовов функции невязки: {result.nfev}")
    rss = np.sum(result.fun ** 2)
    print(f"\nОстаточная сумма квадратов (RSS): {rss:.6e}")

    # --- 4.5. финальная симуляция с подобранным nu -------------------------
    model_energy_final = simulate_ns(
        [nu_fit], vx0, vy0, t_data, n, L, dt_sim
    )

    # --- 4.6. сравнение данные-vs-модель по годам/шагам -------------------
    print("\n  Шаг |   E(данные)   E(модель)   |разность|")
    for i in range(len(t_data)):
        diff = abs(data_energy[i] - model_energy_final[i])
        print(f"  {i:4d} | {data_energy[i]:12.6f} {model_energy_final[i]:12.6f} "
              f"| {diff:10.6f}")

    # --- 4.7. график сравнения --------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(t_data, data_energy, "o-", label="данные (The Well)",
            color="tab:blue", markersize=4)
    ax.plot(t_data, model_energy_final, "s--",
            label=f"модель (nu = {nu_fit:.4f})",
            color="tab:red", markersize=4)

    if nu_true is not None:
        # дополнительно показываем модель с истинной вязкостью
        model_energy_true = simulate_ns(
            [nu_true], vx0, vy0, t_data, n, L, dt_sim
        )
        ax.plot(t_data, model_energy_true, ":", alpha=0.7,
                label=f"модель (nu_true = {nu_true:.4f})",
                color="tab:green")

    ax.set_xlabel("Время t")
    ax.set_ylabel("Кинетическая энергия E(t)")
    ax.set_title("Сравнение данные-vs-модель: подбор вязкости nu (LM)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = os.path.join("outputs", "navier_stokes_lm_fit.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nГрафик сохранён: {out_path}")

    # --- 4.8. дополнительная визуализация: завихренность ------------------
    # сравнение полей завихренности в последний момент времени
    solver = NavierStokes2D(n=n, L=L)
    # финальная модель
    traj_x, traj_y = solver.simulate(
        vx0, vy0, nu=nu_fit, dt=dt_sim,
        n_steps=(len(t_data) - 1) * max(1, int(round(dt_data / dt_sim))),
        save_every=max(1, int(round(dt_data / dt_sim))),
    )
    vx_model, vy_model = traj_x[-1], traj_y[-1]

    # завихренность omega = dvy/dx - dvx/dy
    omega_data = solver.ddx(vy_data[-1]) - solver.ddy(vx_data[-1])
    omega_model = solver.ddx(vy_model) - solver.ddy(vx_model)

    fig2, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    im0 = axes[0].imshow(omega_data, origin="lower", cmap="RdBu_r")
    axes[0].set_title("Завихренность: данные (последний кадр)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(omega_model, origin="lower", cmap="RdBu_r")
    axes[1].set_title(f"Завихренность: модель (nu = {nu_fit:.4f})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    out_path2 = os.path.join("outputs", "navier_stokes_lm_vorticity.png")
    fig2.tight_layout()
    fig2.savefig(out_path2, dpi=150)
    print(f"Сравнение завихренности сохранено: {out_path2}")


if __name__ == "__main__":
    main()