import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import Table
from xgboost import XGBClassifier
import scienceplots  # noqa: F401  (используется стилем matplotlib)
from source import *  # noqa: F403,F401  (используются внешние утилиты)

plt.style.use(['science', 'ieee', 'grid'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
plt.rcParams["figure.figsize"] = (12 / 1.5, 6 / 1.5)

sig = ds.dataset(list(Path("new_sig/").glob("*.parquet")), format="parquet")
bsbs = ds.dataset(list(Path("gen_data/").glob("bsbs*4.parquet")), format="parquet")
nonbsbs = ds.dataset(list(Path("gen_data/").glob("nonbsbs*4.parquet")),
                     format="parquet")
uds = ds.dataset(list(Path("gen_data/").glob("uds*4.parquet")), format="parquet")

scanner = sig.scanner(batch_size=100_000)
for batch in scanner.to_batches():
    table = Table.from_batches([batch])
    break
del scanner

mask = (
    (pc.field("E_gamma_in_ROE") < 1.2)
    & (pc.field("Bs_lik") > 0.0012)
    & (pc.field("N_KS") < 0.5)
)

mask_is1 = (
    pc.field("correct_nu_0")
    & pc.field("correct_nu_1")
    & pc.field("correct_gamma_0")
    & pc.field("correct_gamma_1")
    & pc.field("lost_0")
    & pc.field("lost_1")
    & pc.field("Miss_id_0")
    & pc.field("Miss_id_1")
)

mask_is0 = pc.field("is0") == 1

mask_lep = (
    (pc.field("idec0").isin([0, 1]))
    & (pc.field("idec1").isin([0, 1]))
)
mask_pi = ((pc.field("idec0") == 2) & (pc.field("idec0") == 2))
mask_rho = ((pc.field("idec0") == 3) & (pc.field("idec0") == 3))
mask_pirho = (
    (pc.field("idec0") == 2) & (pc.field("idec1") == 3)
) | (
    (pc.field("idec0") == 3) & (pc.field("idec1") == 2)
)
mask_pi_lep = (
    ((pc.field("idec0") == 2) & pc.field("idec1").isin([0, 1]))
    | ((pc.field("idec1") == 2) & pc.field("idec0").isin([0, 1]))
)
mask_rho_lep = (
    ((pc.field("idec0") == 3) & pc.field("idec1").isin([0, 1]))
    | ((pc.field("idec1") == 3) & pc.field("idec0").isin([0, 1]))
)

mods_cut = {
    'lep': [mask_lep, 'lep', 'e / \\mu'],
    'pipi': [mask_pi, 'pirho', '\\pi \\pi'],
    'rhorho': [mask_rho, 'pirho', '\\rho \\rho'],
    'pirho': [mask_pirho, 'pirho', '\\pi \\rho'],
    'pilep': [mask_pi_lep, 'pirho_lep', '\\pi \\ell'],
    'rholep': [mask_rho_lep, 'pirho_lep', '\\rho \\ell'],
}

if len(sys.argv) > 1:
    current_mod = sys.argv[1]
    use_BDT = sys.argv[2]
else:
    current_mod = 'pipi'
    use_BDT = 'n'

sig = get_values(  # noqa: F405
    sig,
    ['E_gamma_in_ROE', 'MBtag'],
    filter_mask=(
        mask & mods_cut[current_mod][0] & mask_is1 & (pc.field("is0") == 1)
    ),
)

model = XGBClassifier()
model.load_model(f"models/bdt_model_{mods_cut[current_mod][1]}.json")

import json  # после model.load_model, чтобы не ломать порядок импорта выше

with open(f"models/columns_and_fom_{mods_cut[current_mod][1]}.json", "r") as f:
    data_to_save = json.load(f)

columns = data_to_save["columns"]
if use_BDT == 'y':
    FoM = data_to_save["FoM"][current_mod]
else:
    FoM = 0
print(f"FoM = {FoM}")

common_cols = columns + ['E_gamma_in_ROE', 'MBtag']

v1 = get_values(bsbs, common_cols,
                filter_mask=(mask & mods_cut[current_mod][0]
                             & (pc.field("is0") == 1)))
v2 = get_values(nonbsbs, common_cols,
                filter_mask=(mask & mods_cut[current_mod][0]
                             & (pc.field("is0") == 1)))
v3 = get_values(uds, common_cols,
                filter_mask=(mask & mods_cut[current_mod][0]
                             & (pc.field("is0") == 1)))
v4 = get_values(uds, common_cols,
                filter_mask=(mask & mods_cut[current_mod][0]
                             & (pc.field("is0") == 0)))
v5 = get_values(nonbsbs, common_cols,
                filter_mask=(mask & mods_cut[current_mod][0]
                             & (pc.field("is0") == 0)))
v6 = get_values(bsbs, common_cols,
                filter_mask=(mask & mods_cut[current_mod][0]
                             & (pc.field("is0") == 0)))

v1["marker"] = 1  # $B_s \to D(\ell \nu)\ell \nu$
v2["marker"] = 2  # nonbsbs
v3["marker"] = 3  # continuum
v4["marker"] = 4  # B_s^{tag} \ rec. \ err.
v5["marker"] = 5  # B_s^{tag} \ rec. \ err.
v6["marker"] = 6  # B_s^{tag} \ rec. \ err.

data = pd.concat([v1, v2, v3, v4, v5, v6], ignore_index=True)

X_new = data[columns]
bdt_scores = model.predict_proba(X_new)[:, 1]
data["bdt_score"] = bdt_scores

selected_mask = data["bdt_score"] > FoM
selected_data = data[selected_mask]

# --------- E_gamma_in_ROE -----------
wbin = 0.1
if use_BDT == 'y':
    wbin = wbin * 1.5
a, b = 0.0, 1.2
bins = np.linspace(a, b, int((b - a) // wbin) + 2)

marker_labels = {
    1: r"$B_s$ wrong sig",
    2: "nonbsbs",
    3: "Continuum",
    4: r"$B_s$ wrong tag uds",
    5: r"$B_s$ wrong tag nonbsbs",
    6: r"$B_s$ wrong tag bsbs",
}

stack_data = []
stack_labels = []
for marker in sorted(
    selected_data["marker"].unique(),
    key=lambda x: (selected_data["marker"] == x).sum(),
):
    subset = selected_data[selected_data["marker"] == marker]["E_gamma_in_ROE"]
    stack_data.append(subset)
    stack_labels.append(f"{marker_labels[marker]} $N = {len(subset)}$")

Ntot = sum(len(sub) for sub in stack_data)
weights = [np.full(len(sub), 1.0 / max(Ntot, 1)) for sub in stack_data]

plt.hist(
    stack_data,
    bins=bins,
    stacked=True,
    weights=weights,
    alpha=0.55,
    edgecolor="black",
    linewidth=1.5,
    label=stack_labels,
)

errorhist(  # noqa: F405
    sig["E_gamma_in_ROE"],
    bins=bins,
    density=True,
    label='signal',
)

plt.xlabel(r"$E_\gamma^{\mathrm{ROE}},\ \mathrm{GeV}$")
plt.ylabel(fr'$\mathrm{{Events}}\,/\,{wbin}\,\mathrm{{GeV}}$')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), framealpha=0.85)
plt.savefig(f"output/stack/bg_E_ROE_stack_{current_mod}_{use_BDT}.pdf", dpi=700)
plt.clf()

# ------------- MBtag -----------------
Bs_m = 5.36693
wbin = 0.01
if use_BDT == 'y':
    wbin = wbin * 3
a, b = Bs_m - 0.11, Bs_m + 0.11
bins = np.linspace(a, b, int((b - a) // wbin) + 2)

marker_labels = {
    1: r"$B_s$ wrong sig",
    2: "nonbsbs",
    3: "Continuum",
    4: r"$B_s$ wrong tag uds",
    5: r"$B_s$ wrong tag nonbsbs",
    6: r"$B_s$ wrong tag bsbs",
}

stack_data = []
stack_labels = []
for marker in sorted(
    selected_data["marker"].unique(),
    key=lambda x: (selected_data["marker"] == x).sum(),
):
    subset = selected_data[selected_data["marker"] == marker]["MBtag"]
    stack_data.append(subset)
    stack_labels.append(f"{marker_labels[marker]} $N = {len(subset)}$")


Ntot = sum(len(sub) for sub in stack_data)
weights = [np.full(len(sub), 1.0 / max(Ntot, 1)) for sub in stack_data]

plt.hist(
    stack_data,
    bins=bins,
    stacked=True,
    weights=weights,
    alpha=0.55,
    edgecolor="black",
    linewidth=1.5,
    label=stack_labels,
)
errorhist(  # noqa: F405
    sig["MBtag"],
    bins=bins,
    density=True,
    label='signal',
)

plt.xlabel(r"$M_{B_s^0},\ \mathrm{GeV}$")
plt.ylabel(fr'$\mathrm{{Events}}\,/\,{wbin}\,\mathrm{{GeV}}$')
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), framealpha=0.85)
plt.savefig(f"output/stack/bg_M_BS0_stack_{current_mod}_{use_BDT}.pdf", dpi=700)
plt.clf()
