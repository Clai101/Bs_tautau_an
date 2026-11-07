"""Общие импорты и составные маски для BDT и построения форм сигнала/фона.

Маски сигнала и фона собираются в ноутбуках и скриптах из заданных здесь частей.
Коды распадов τ: 0 — e, 1 — μ, 2 — π, 3 — ρ.
"""

# Импорты анализа и обучения
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scienceplots  # Регистрирует стили matplotlib.
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from source import colors, errordot, get_values, hist_approx, json_update, normalization

plt.style.use(["science", "ieee", "grid"])

# Масса Bs в GeV, как в header.py (для осей диагностических графиков).
Bs_m = 5.36693


# Единственный источник палитры — source.colors: k, r, b, g.
# k с alpha=0.55 даёт серую заливку, как в исходных stack-гистограммах.
PLOT_COLORS = {
    "signal": colors[1],
    "bsbs": colors[3],
    "bsbs_other": colors[0],
    "nonbsbs": colors[0],
    "uds_charm": colors[0],
    "ubs": colors[0],
    "background": colors[0],
    "data": colors[0],
    "diagnostic": colors[2],
}
PLOT_CMAPS = {
    name: LinearSegmentedColormap.from_list(
        f"analysis_{name}", [plt.rcParams["axes.facecolor"], color],
    )
    for name, color in PLOT_COLORS.items()
}
BACKGROUND_SOURCE_ORDER = ("uds_charm", "nonbsbs", "bsbs_other", "bsbs")
BACKGROUND_LABELS = {
    "uds_charm": "Continuum (uds/charm)",
    "nonbsbs": "non-BsBs",
    "bsbs_other": "BsBs, wrong/undefined tag",
    "bsbs": "BsBs, correct tag",
    "ubs": "Other backgrounds (ubs)",
}
# Прежняя подпись Bs -> D(lnu)lnu относится к категории bsbs; is0 == 1
# проверяет только тег и сам по себе не подтверждает эту цепочку распада.
BACKGROUND_FIT_GROUPS = {
    "bsbs": ("bsbs",),
    "ubs": ("uds_charm", "nonbsbs", "bsbs_other"),
}


def classify_background_source(filename, is0):
    """Четыре непересекающихся источника для проекций и сборки PDF фона."""
    name = Path(filename).name
    if name.startswith("bsbs_"):
        correct_tag = is0.eq(1).fillna(False).to_numpy(dtype=bool)
        return np.where(correct_tag, "bsbs", "bsbs_other")
    if name.startswith("nonbsbs_"):
        return np.full(len(is0), "nonbsbs")
    if name.startswith(("uds", "charm")):
        return np.full(len(is0), "uds_charm")
    raise ValueError(f"Неизвестный источник фонового MC: {name}")


# 1. Отбор по наблюдаемым величинам (без MC truth)
mask_energy = pc.field("E_gamma_in_ROE") < 1.2
mask_tag_quality = pc.field("Bs_lik") > 0.0012
mask_ks_veto = pc.field("N_KS") < 0.5
mask_physics = mask_energy & mask_tag_quality & mask_ks_veto


# 2. Проверки MC-информации, отдельно для каждого типа проверки
mask_mc_neutrinos = pc.field("correct_nu_0") & pc.field("correct_nu_1")
mask_mc_photons = pc.field("correct_gamma_0") & pc.field("correct_gamma_1")
mask_mc_lost = pc.field("lost_0") & pc.field("lost_1")
mask_mc_pid = pc.field("Miss_id_0") & pc.field("Miss_id_1")
mask_mc_signal = mask_mc_neutrinos & mask_mc_photons & mask_mc_lost & mask_mc_pid
mask_mc_tag = pc.field("is0") == 1
mask_mc_wrong_tag = pc.field("is0") == 0
# Значения/полярность lost и Miss_id сохранены как в исходных ноутбуках.


# 3. Каналы из элементарных масок двух τ
def decay_channel_masks(tau0, tau1):
    """Одинаковые определения для Arrow-полей и pandas.Series из X_test."""
    tau0_lep, tau1_lep = tau0.isin([0, 1]), tau1.isin([0, 1])
    tau0_pi, tau1_pi = tau0 == 2, tau1 == 2
    tau0_rho, tau1_rho = tau0 == 3, tau1 == 3
    return {
        "lep": tau0_lep & tau1_lep,
        "pipi": tau0_pi & tau1_pi,
        "rhorho": tau0_rho & tau1_rho,
        "pirho": (tau0_pi & tau1_rho) | (tau0_rho & tau1_pi),
        "pilep": (tau0_pi & tau1_lep) | (tau0_lep & tau1_pi),
        "rholep": (tau0_rho & tau1_lep) | (tau0_lep & tau1_rho),
    }


channel_selections = decay_channel_masks(pc.field("idec0"), pc.field("idec1"))
mask_lep = channel_selections["lep"]
mask_pipi = channel_selections["pipi"]
mask_rhorho = channel_selections["rhorho"]
mask_pirho = channel_selections["pirho"]
mask_pilep = channel_selections["pilep"]
mask_rholep = channel_selections["rholep"]

# Модели и подписи общих каналов BDT и построения форм.
model_groups = {
    "lep": "lep", "pipi": "pirho", "rhorho": "pirho", "pirho": "pirho",
    "pilep": "pirho_lep", "rholep": "pirho_lep",
}
channel_labels = {
    "lep": r"\ell\ell", "pipi": r"\pi\pi", "rhorho": r"\rho\rho",
    "pirho": r"\pi\rho", "pilep": r"\pi\ell", "rholep": r"\rho\ell",
}
# Имена в JSON смешанного BDT отличаются от имён физических каналов.
cut_aliases = {"pilep": "pi_ell", "rholep": "rho_ell"}


# Явный список имён, доступных для импорта в ноутбуки и скрипты.
__all__ = [
    "argparse", "subprocess", "sys", "json", "Path", "np", "pd", "pc", "ds", "plt", "colors",
    "classification_report", "confusion_matrix", "roc_auc_score",
    "train_test_split", "XGBClassifier", "errordot", "get_values", "hist_approx", "json_update",
    "normalization", "Bs_m",
    "PLOT_COLORS", "PLOT_CMAPS", "BACKGROUND_SOURCE_ORDER", "BACKGROUND_LABELS",
    "BACKGROUND_FIT_GROUPS", "classify_background_source",
    "mask_energy", "mask_tag_quality", "mask_ks_veto", "mask_physics",
    "mask_mc_neutrinos", "mask_mc_photons", "mask_mc_lost", "mask_mc_pid",
    "mask_mc_signal", "mask_mc_tag", "mask_mc_wrong_tag", "decay_channel_masks", "channel_selections",
    "mask_lep", "mask_pipi", "mask_rhorho", "mask_pirho", "mask_pilep", "mask_rholep",
    "model_groups", "channel_labels", "cut_aliases",
]
