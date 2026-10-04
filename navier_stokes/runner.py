"""
Единый раннер для папки navier_stokes.

Запускает все скрипты пайплайна в правильном порядке, кроме
prepare_well_slice.py (он требует интернет и запускается вручную на
локальной машине один раз; предполагается, что well_shear_flow_slice.npz
уже лежит рядом).

Порядок и обоснование:
  1. navier_stokes_tree.py      — строит DAG уравнения (структурная основа,
                                  без численных расчётов).
  2. navier_stokes_solver.py    — численный псевдоспектральный решатель
                                  (проверочный прогон + вывод E(t)).
  3. visualize_navier_stokes.py — граф дерева + картина завихренности,
                                  сохраняет PNG в outputs/.
  4. fit_navier_stokes_lm.py    — подгонка вязкости nu методом
                                  Левенберга-Марквардта на реальной
                                  траектории из The Well + финальное
                                  сравнение данные-vs-модель.

Каждый скрипт запускается как отдельный процесс (subprocess), чтобы
изолировать глобальное состояние numpy/matplotlib и не ловить побочные
эффекты от последовательных импортов. При падении любого шага раннер
останавливается и печатает диагностику.

Использование:
    python run_all_navier_stokes.py
    python run_all_navier_stokes.py --skip-fit     # пропустить шаг 4
    python run_all_navier_stokes.py --only solver  # запустить только один шаг
"""

import argparse
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Список шагов пайплайна: (короткое имя, файл, описание)
# ---------------------------------------------------------------------------
STEPS = [
    ("tree",     "navier_stokes_tree.py",      "Построение DAG уравнения Навье-Стокса"),
    ("solver",   "navier_stokes_solver.py",    "Численный прогон псевдоспектрального решателя"),
    ("visualize", "visualize_navier_stokes.py", "Визуализация: граф дерева + завихренность"),
    ("fit",      "fit_navier_stokes_lm.py",     "Подгонка nu (Levenberg-Marquardt) + данные-vs-модель"),
]

# Файл, который НЕ запускаем автоматически — он требует интернет
EXCLUDED = {"prepare_well_slice.py"}


def run_step(name, filename, extra_env=None):
    """Запускает один python-файл как подпроцесс в текущей директории."""
    if filename in EXCLUDED:
        print(f"[SKIP] {filename} — исключён из автозапуска (требует интернет).")
        return True

    if not os.path.exists(filename):
        print(f"[FAIL] {filename} не найден в текущей директории.")
        return False

    print("\n" + "=" * 72)
    print(f"[STEP:{name}] Запуск {filename}")
    print("=" * 72)

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    t0 = time.time()
    try:
        # sys.executable — тот же интерпретатор, что и у раннера
        result = subprocess.run(
            [sys.executable, filename],
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        print(f"\n[ABORT] Прервано пользователем на шаге '{name}'.")
        return False

    dt = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL] {filename} завершился с кодом {result.returncode} "
              f"за {dt:.1f} с.")
        return False

    print(f"[OK]   {filename} успешно завершён за {dt:.1f} с.")
    return True


def ensure_data_present():
    """
    Проверяет, что срез The Well уже лежит рядом. Если нет — предупреждает,
    но не падает: шаги 1–3 можно выполнить и без данных, шаг 4 тогда
    корректно завершится с понятной ошибкой.
    """
    path = "well_shear_flow_slice.npz"
    if not os.path.exists(path):
        print(f"[WARN] Файл {path} не найден. Шаги 1–3 выполнятся, "
              f"а шаг 4 (подгонка) потребует его наличия.\n"
              f"       Запустите prepare_well_slice.py вручную на машине "
              f"с интернетом и положите .npz рядом со скриптами.\n")
    else:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[INFO] Найден {path} ({size_mb:.2f} МБ).")


def main():
    parser = argparse.ArgumentParser(
        description="Единый раннер для папки navier_stokes."
    )
    parser.add_argument(
        "--only", type=str, default=None,
        choices=[s[0] for s in STEPS],
        help="Запустить только один указанный шаг (по короткому имени).",
    )
    parser.add_argument(
        "--skip-fit", action="store_true",
        help="Пропустить шаг 4 (подгонка nu).",
    )
    parser.add_argument(
        "--skip-vis", action="store_true",
        help="Пропустить шаг 3 (визуализация).",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Запуск пайплайна navier_stokes")
    print("=" * 72)

    ensure_data_present()

    # --- отбор шагов к запуску --------------------------------------------
    if args.only is not None:
        steps_to_run = [s for s in STEPS if s[0] == args.only]
        if not steps_to_run:
            print(f"[FAIL] Неизвестный шаг: {args.only}")
            sys.exit(1)
    else:
        steps_to_run = list(STEPS)
        if args.skip_vis:
            steps_to_run = [s for s in steps_to_run if s[0] != "visualize"]
        if args.skip_fit:
            steps_to_run = [s for s in steps_to_run if s[0] != "fit"]

    print(f"\nЗапланированные шаги: {[s[0] for s in steps_to_run]}\n")

    # --- последовательный запуск ------------------------------------------
    t_start = time.time()
    for name, filename, description in steps_to_run:
        print(f"\n>>> {name}: {description}")
        ok = run_step(name, filename)
        if not ok:
            print(f"\n[ABORT] Пайплайн остановлен на шаге '{name}'.")
            sys.exit(1)

    total = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"Пайплайн завершён успешно. Общее время: {total:.1f} с.")
    print(f"Артефакты: outputs/navier_stokes_tree_graph.png, "
          f"outputs/navier_stokes_vorticity_evolution.png, "
          f"outputs/navier_stokes_lm_fit.png, "
          f"outputs/navier_stokes_lm_vorticity.png")
    print("=" * 72)


if __name__ == "__main__":
    main()