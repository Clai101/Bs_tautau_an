import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pathlib import Path
import json
from source import *

mask = (pc.field("E_gamma_in_ROE") < 1.2) & (pc.field("Bs_lik") > 0.0012) & (pc.field("N_KS") < 0.5)
mask_M = (pc.field("M0") > 5.347) &  (pc.field("M0") < 5.387) 
mask_is1 = pc.field("correct_nu_0") & pc.field("correct_nu_1") & pc.field("correct_gamma_0") & pc.field("correct_gamma_1") & pc.field("lost_0") & pc.field("lost_1") & pc.field("Miss_id_0") & pc.field("Miss_id_1")
mask_is0 = pc.field("is0") == 1
mask_miss_id = pc.field("Miss_id_0") & pc.field("Miss_id_1")
mask_lost_Kpi = pc.field("lost_0") & pc.field("lost_1")

mask_lep = ((pc.field("idec0") == 0) | (pc.field("idec0") == 1)) & ((pc.field("idec1") == 0) | (pc.field("idec1") == 1))
mask_pi = ((pc.field("idec0") == 2) & (pc.field("idec0") == 2))
mask_rho = ((pc.field("idec0") == 3) & (pc.field("idec0") == 3))
mask_pirho = ((pc.field("idec0") == 2) & (pc.field("idec1") == 3)) | ((pc.field("idec0") == 3) & (pc.field("idec1") == 2))
mask_pilep = ((pc.field("idec0") == 2)) & ((pc.field("idec1") == 0) | (pc.field("idec1") == 1)) | ((pc.field("idec1") == 2)) & ((pc.field("idec0") == 0) | (pc.field("idec0") == 1))
mask_rholep = ((pc.field("idec0") == 3)) & ((pc.field("idec1") == 0) | (pc.field("idec1") == 1)) | ((pc.field("idec1") == 3)) & ((pc.field("idec0") == 0) | (pc.field("idec0") == 1))

#decay_mod_tau = ["$e^+ \\nu_e \\bar \\nu_\\tau$", "$\mu^+ \\nu_\\mu \\bar \\nu_\\tau$", "$\\pi^+ \\bar \\nu_\\tau$", "$\\rho^+ (\\pi^+ \\pi^0) \\bar \\nu_\\tau$", "$\\pi^+ \\pi^+ \\pi^- \\bar \\nu_\\tau$", "$\\rho^+ (\\pi^+ \\gamma) \\bar \\nu_\\tau$"]

mask_sig = (pc.field("is0") == 1) & mask & mask_is1
mask_bkg = mask


Lamc_m = 2.28646
Lamc_25_m = 2.5925
Lamc_26_m = 2.628
Lam_m = 1.115683

D_0_m = 1.86483
D_p_m = 1.86966

D_st_p_m = 2.01026
D_st_0_m = 2.00685

Pi_p_m = 0.13957
Pi_0_m = 0.13498

D_st_D_dif = 0.142014
D_s_m = 1.96835

K_s_m = 0.497611
K_p_m = 0.493677

Bs_m = 5.36693

tau_m = 1.77693
mu_m = 0.1056583755


