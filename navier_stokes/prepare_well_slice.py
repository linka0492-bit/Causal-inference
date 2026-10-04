"""
ЗАПУСКАТЬ НА ВАШЕЙ МАШИНЕ (не в песочнице) -- там, где есть интернет.

Устанавливает пакет the_well, скачивает маленький срез датасета
"shear_flow" (2D несжимаемые уравнения Навье-Стокса, периодическая
область -- канонический Navier-Stokes датасет в The Well, в отличие
от MHD, где есть ещё магнитное поле) и сохраняет его в .npz,
который можно будет просто загрузить в чат.

Установка перед запуском:
    pip install the_well h5py

Если имя датасета/API изменилось относительно того, что здесь
написано (the_well активно развивается) -- посмотрите актуальный
README: https://github.com/PolymathicAI/the_well
и поправьте здесь имя датасета / способ загрузки; сама логика
сохранения в .npz внизу не изменится.
"""

import numpy as np

def main():
    from the_well.data import WellDataset

    # "shear_flow" -- 2D несжимаемые уравнения Навье-Стокса с периодическими
    # граничными условиями, несколько траекторий с разными числами Рейнольдса.
    # В текущем виде датасет лежит напрямую в организации polymathic-ai на HF,
    # поэтому base path должен указывать на корень репозитория организации.
    ds = WellDataset(
        well_base_path="hf://datasets/polymathic-ai/",
        well_dataset_name="shear_flow",
        well_split_name="test",
    )

    print(f"Датасет содержит {len(ds)} траекторий/сэмплов")
    sample = ds[0]

    print("\nКлючи в одном сэмпле:")
    for k, v in sample.items():
        shape = getattr(v, "shape", None)
        print(f"  {k}: {type(v)}  shape={shape}")

    # Сохраняем ВСЁ, что нашли, как есть -- дальше на стороне анализа
    # разберёмся, что из этого density/velocity/dt/dx
    to_save = {}
    for k, v in sample.items():
        try:
            to_save[k] = np.asarray(v)
        except Exception as e:
            print(f"  (пропускаю ключ {k}: {e})")

    out_path = "well_shear_flow_slice.npz"
    np.savez_compressed(out_path, **to_save)
    print(f"\nСохранено: {out_path}")
    print("Загрузите этот файл в чат.")


if __name__ == "__main__":
    main()
