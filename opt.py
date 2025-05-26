import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path(".")
output_dir = data_dir / "optimized"
output_dir.mkdir(exist_ok=True)

int8_cols = [
    "idec0", "idec1", "N_KL", "is0"
]

to_drop = [
    "lost_gamma_0", "lost_pi_0", "lost_K_0",
    "lost_gamma_1", "lost_pi_1", "lost_K_1",
    "Miss_id_0", "Miss_id_1", "N_tracks_in_ROE", "lost_nu_0", "lost_nu_1",
]

bool_cols = ["Miss_id_0", "Miss_id_1"]

int32_cols = ["__experiment__", "__run__", "__event__"]

float32_cols = [
    "missedE", "M0", "p0", "recM2", "totalEnergyMC", "E_gamma_in_ROE", "Bs_lik"
]

decay_mode_to_nu = {
    0: 2,  # e+ nu_e nu_tau
    1: 2,  # mu+ nu_mu nu_tau
    2: 1,  # pi+ nu_tau
    3: 1,  # rho+ (pi+ pi0) nu_tau
    4: 1,  # pi+ pi+ pi- nu_tau
    5: 1
}
decay_mode_to_gamma = {
    0: 0,  # e+ nu_e nu_tau
    1: 0,  # mu+ nu_mu nu_tau
    2: 0,  # pi+ nu_tau
    3: 0,  # rho+ (pi+ pi0) nu_tau
    4: 0,  # pi+ pi+ pi- nu_tau
    5: 1   # rho+ (pi+ gamma) nu_tau
}

def safe_downcast(col, target_type):
    try:
        return pd.to_numeric(col, downcast=target_type)
    except Exception as e:
        print(f"Ошибка при преобразовании {col.name} → {target_type}: {e}")
        return col

for path in data_dir.glob("*.feather"):
    print(f"\nОбработка файла: {path.name}")
    df = pd.read_feather(path)

    for idx in [0, 1]:
        df["correct_nu_0"] = (df['lost_nu_0'] == df['idec0'].map(decay_mode_to_nu))
        df["correct_nu_1"] = (df['lost_nu_1'] == df['idec1'].map(decay_mode_to_nu))
        df["correct_gamma_0"] = (df['lost_gamma_0'] == df['idec0'].map(decay_mode_to_gamma))
        df["correct_gamma_1"] = (df['lost_gamma_1'] == df['idec1'].map(decay_mode_to_gamma))
        df["lost_0"] = ((df['lost_K_0'] == 0) & (df['lost_pi_0'] == 0))
        df["lost_1"] = ((df['lost_K_1'] == 0) & (df['lost_pi_1'] == 0))

    for col in df.columns:
        if col in bool_cols:
            df[col] = df[col].astype("boolean")
        elif col in int8_cols:
            df[col] = safe_downcast(df[col], "unsigned")
        elif col in int32_cols:
            df[col] = safe_downcast(df[col], "unsigned")
        elif col in float32_cols:
            df[col] = safe_downcast(df[col], "float")
        elif col in to_drop:
            df.drop(columns=col, inplace=True)
        else:
            df[col] = safe_downcast(df[col], "float")

    out_path = output_dir / path.name
    df.reset_index(drop=True, inplace=True)
    df.to_feather(out_path)

    print(f"Сохранено: {out_path.name}")

print("\nВсе файлы обработаны. Оптимизированные версии находятся в папке 'optimized/'")
