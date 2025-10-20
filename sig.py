import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as dsF
import pyarrow.compute as pc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
import json
from source import *
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import sys
import scienceplots

plt.style.use(['science','ieee', 'grid'])
dataset = ds.dataset(list(Path(f"new_sig/").glob("*.parquet")), format="parquet")
scanner = dataset.scanner(batch_size=100_000)
for batch in scanner.to_batches():
    table = pa.Table.from_batches([batch])
    print(table.to_pandas().columns)
    break
del scanner
decay_mod_tau = ["$e^+ \\nu_e \\bar \\nu_\\tau$", "$\mu^+ \\nu_\\mu \\bar \\nu_\\tau$", "$\\pi^+ \\bar \\nu_\\tau$", "$\\rho^+ (\\pi^+ \\pi^0) \\bar \\nu_\\tau$", "$\\pi^+ \\pi^+ \\pi^- \\bar \\nu_\\tau$", "$\\rho^+ (\\pi^+ \\gamma) \\bar \\nu_\\tau$"]
mask = (pc.field("E_gamma_in_ROE") < 1.2) & (pc.field("Bs_lik") > 0.0012) & (pc.field("N_KS") < 0.5)
mask_is1 = pc.field("correct_nu_0") & pc.field("correct_nu_1") & pc.field("correct_gamma_0") & pc.field("correct_gamma_1") & pc.field("lost_0") & pc.field("lost_1") & pc.field("Miss_id_0") & pc.field("Miss_id_1")
mask_is0 = pc.field("is0") == 1
mask_sig = (pc.field("is0") == 1) & mask & mask_is1
mask_lep = ((pc.field("idec0") == 0) | (pc.field("idec0") == 1)) & ((pc.field("idec1") == 0) | (pc.field("idec1") == 1))
mask_pi = ((pc.field("idec0") == 2) & (pc.field("idec0") == 2))
mask_rho = ((pc.field("idec0") == 3) & (pc.field("idec0") == 3))
mask_pirho = ((pc.field("idec0") == 2) & (pc.field("idec1") == 3)) | ((pc.field("idec0") == 3) & (pc.field("idec1") == 2))
mask_pi_lep = ((pc.field("idec0") == 2)) & ((pc.field("idec1") == 0) | (pc.field("idec1") == 1)) | ((pc.field("idec0") == 0) | (pc.field("idec0") == 1)) & ((pc.field("idec1") == 2)) 
mask_rho_lep = ((pc.field("idec0") == 3)) & ((pc.field("idec1") == 0) | (pc.field("idec1") == 1)) | ((pc.field("idec0") == 0) | (pc.field("idec0") == 1)) & ((pc.field("idec1") == 3))
decay_mod_tau = ["$e^+ \\nu_e \\bar \\nu_\\tau$", "$\mu^+ \\nu_\\mu \\bar \\nu_\\tau$", "$\\pi^+ \\bar \\nu_\\tau$", "$\\rho^+ (\\pi^+ \\pi^0) \\bar \\nu_\\tau$", "$\\pi^+ \\pi^+ \\pi^- \\bar \\nu_\\tau$", "$\\rho^+ (\\pi^+ \\gamma) \\bar \\nu_\\tau$"]
mods_cut = {'lep': [mask_lep, 'lep', 'e / \\mu'],
        'pipi': [mask_pi, 'pirho', '\\pi \\pi'],
        'rhorho': [mask_rho, 'pirho', '\\rho \\rho'],
        'pirho': [mask_pirho, 'pirho', '\\pi \\rho'],
        'pilep': [mask_pi_lep, 'pirho_lep', '\\pi \\ell'],
        'rholep': [mask_rho_lep, 'pirho_lep', '\\rho \\ell']}

if len(sys.argv) > 1:
    current_mod = sys.argv[1]

import json
with open(f"models/columns_and_fom_{mods_cut[current_mod][1]}.json", "r") as f:
    data_to_save = json.load(f)
columns = data_to_save["columns"]
FoM = data_to_save["FoM"]
data = get_values(dataset, columns + ['E_gamma_in_ROE', 'MBtag'], filter_mask=((pc.field("is0") == 1) & mask & mask_is1 & mods_cut[current_mod][0]))
from xgboost import XGBClassifier
import json
model = XGBClassifier()
model.load_model(f"models/bdt_model_{mods_cut[current_mod][1]}.json")
with open(f"models/columns_and_fom_{mods_cut[current_mod][1]}.json", "r") as f:
    data_to_save = json.load(f)
columns = data_to_save["columns"]
FoM = data_to_save["FoM"][current_mod]
X_new = data[columns]

bdt_scores = model.predict_proba(X_new)[:, 1] 

data["bdt_score"] = bdt_scores

selected_mask = data["bdt_score"] > FoM

selected_data = data[selected_mask]
wbin = 0.05
a = 0 
b = 1.2
bins = np.linspace(a, b, int((b-a)/wbin))
BS0 = 5.366691
wbin_x, a_x, b_x = 0.06, 0, 1.2
wbin_y, a_y, b_y = 0.0075, 5.25, 5.5

bins_x = np.linspace(a_x, b_x, int((b_x - a_x) // wbin_x + 2))
bins_y = np.linspace(a_y, b_y, int((b_y - a_y) // wbin_y + 2))
values = selected_data[["E_gamma_in_ROE", "MBtag"]].values

counts, _ = np.histogramdd(values, bins=(bins_x, bins_y))
bin_centers = [0.5 * (b[:-1] + b[1:]) for b in (bins_x, bins_y)]
bin_centers = np.meshgrid(*bin_centers, indexing='ij')

plt.figure(figsize=(8, 6))
pmesh = plt.pcolormesh(*bin_centers, counts)
plt.xlabel(r"$E_\gamma^{\mathrm{ROE}}\ \mathrm{GeV}$")
plt.ylabel(r"$M_{B_s^0}\ \mathrm{GeV}$")
plt.title(rf"Signal $B_s^0 \to \tau \tau \to {mods_cut[current_mod][2]}$")
plt.colorbar(pmesh, label="Counts")
plt.savefig(f"output/sig_E_ROE_M_Bs{current_mod}.pdf", dpi=700)
plt.show()
hist = hist_approx()
hist.put_hist(bin_centers, counts)
x = np.linspace(0, 1.2, 100)
y = np.linspace(BS0 - 0.2, BS0 + 0.2, 100)
X, Y = np.meshgrid(x, y, indexing='ij')
val = hist.get_counts([X,Y])

plt.figure(figsize=(8, 6))
pmesh = plt.pcolormesh(X, Y, val)
plt.xlabel(r"$E_\gamma^{\mathrm{ROE}}$")
plt.ylabel(r"$M_{B_s^0}$")
plt.colorbar(pmesh, label="Counts")
plt.show()
hist.save_hist(f"shapes/sig_{current_mod}.json")
json_update(f"quantity/N_{current_mod}.json", {"N_sig": selected_data.shape[0]})