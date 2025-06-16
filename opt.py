import pandas as pd
import numpy as np
import uproot
from pathlib import Path

# === Параметры ===
data_dir = Path("..")
output_dir = Path("optimized")
output_dir.mkdir(exist_ok=True)
tree_name = "Y5S"

# === Алиасы и переменные ===
int8_cols = ["idec0", "idec1", "N_KL", "is0"]
int32_cols = ["__experiment__", "__run__", "__event__"]
bool_cols = ["Miss_id_0", "Miss_id_1"]

decay_mode_to_nu = {0: 2, 1: 2, 2: 1, 3: 1, 4: 1, 5: 1}
decay_mode_to_gamma = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
decay_mode_PID = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2}

# Динамически собираем имена переменных
values = []
for tau_ind in [0, 1]:
    for hypo1 in [0, 1, 2, 4]:
        for hypo2 in [0, 1, 2, 4]:
            values.append(f'PID_{hypo1}_vs_{hypo2}_tau{tau_ind}')

to_drop = [
    "lost_gamma_0", "lost_pi_0", "lost_K_0",
    "lost_gamma_1", "lost_pi_1", "lost_K_1",
    "Miss_id_0", "Miss_id_1", "N_tracks_in_ROE", "lost_nu_0", "lost_nu_1"
] + values

def get_all_branches(path, tree_name="Y5S"):
    try:
        with uproot.open(path) as file:
            if tree_name not in file:
                print(f"⚠️ Нет дерева {tree_name} в {path.name}")
                return []
            return list(file[tree_name].keys())
    except Exception as e:
        print(f"❌ Ошибка при чтении {path.name}: {e}")
        return []

# Получаем список всех переменных по первому найденному ROOT-файлу
first_root_file = next(data_dir.rglob("*.root"), None)
if first_root_file is None:
    raise FileNotFoundError("❌ .root-файлы не найдены!")

all_vars = get_all_branches(first_root_file)

# === Функция безопасного преобразования ===
def safe_downcast(col, target_type):
    try:
        return pd.to_numeric(col, downcast=target_type)
    except Exception as e:
        print(f"⚠️ Ошибка при преобразовании {col.name} → {target_type}: {e}")
        return col

# === Обработка одного файла ===
def process_root_file(path):
    try:
        with uproot.open(path) as file:
            if tree_name not in file:
                print(f"⚠️ Нет дерева {tree_name} в {path.name}")
                return
            data = file[tree_name].arrays(all_vars, library="pd")
    except Exception as e:
        print(f"❌ Ошибка чтения {path.name}: {e}")
        return

    df = pd.DataFrame(data)
    df = df[(df["Bs_lik"] > 0.0001) & (np.abs(df["p0"] - 0.47) < 0.1)]

    # Добавление флагов корректности
    df["correct_nu_0"] = df["lost_nu_0"] == df["idec0"].map(decay_mode_to_nu)
    df["correct_nu_1"] = df["lost_nu_1"] == df["idec1"].map(decay_mode_to_nu)
    df["correct_gamma_0"] = df["lost_gamma_0"] == df["idec0"].map(decay_mode_to_gamma)
    df["correct_gamma_1"] = df["lost_gamma_1"] == df["idec1"].map(decay_mode_to_gamma)
    df["lost_0"] = (df["lost_K_0"] == 0) & (df["lost_pi_0"] == 0)
    df["lost_1"] = (df["lost_K_1"] == 0) & (df["lost_pi_1"] == 0)

    for tau_ind in [0, 1]:
        for hypo2 in [0, 1, 2, 4]:
            df[f"PID_self_vs_{hypo2}_tau{tau_ind}"] = 0.0

    for tau_ind in [0, 1]:
        for hypo1 in [0, 1, 2, 4]:
            for hypo2 in [0, 1, 2, 4]:
                name = f"PID_{hypo1}_vs_{hypo2}_tau{tau_ind}"
                mask = df[f"idec{tau_ind}"].map(decay_mode_PID) == hypo1
                df[f"PID_self_vs_{hypo2}_tau{tau_ind}"] += df[name] * mask

    # Типизация и очистка
    for col in df.columns:
        if col in bool_cols:
            df[col] = df[col].astype("boolean")
        elif col in int8_cols or col in int32_cols:
            df[col] = safe_downcast(df[col], "unsigned")
        elif col in to_drop:
            df.drop(columns=col, inplace=True)
        else:
            df[col] = safe_downcast(df[col], "float")

    df.reset_index(drop=True, inplace=True)
    out_path = output_dir / (path.stem + ".parquet")
    df.to_parquet(out_path, index=False)
    print(f"💾 {out_path.name} сохранён")

# === Главный цикл ===
for root_file in data_dir.rglob("*.root"):
    print(f"🔍 Обработка: {root_file.name}")
    process_root_file(root_file)

print("\n🎉 Все файлы обработаны. Результаты — в папке 'optimized/'")
