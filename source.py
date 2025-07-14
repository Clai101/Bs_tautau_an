import numpy as np
import pandas as pd
from pathlib import Path

from scipy.optimize import curve_fit, minimize
from numpy import log, sqrt, exp, pi, e
from matplotlib.axes import Axes

import matplotlib.pyplot as plt
import os
from numba import njit
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

from pyarrow import Table
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

import matplotlib.pyplot as plt

from itertools import cycle

SMALL_SIZE = 20/1.5
MEDIUM_SIZE = 23/1.5
BIGGER_SIZE = 26/1.5


plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('legend', title_fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE) 
plt.rcParams['axes.titlesize'] = BIGGER_SIZE 
plt.rcParams["figure.figsize"] = (12/1.5, 6/1.5) 

import scienceplots
plt.style.use(['science','ieee', 'grid'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

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
K_s_m = 0.497611
K_p_m = 0.493677
Bs_m = 5.36693
tau_m = 1.77693
mu_m = 0.1056583755
D_s_m = 1.96835

x = np.array([1. , 2. ])
n = np.array([1, 2])

@njit(fastmath = True)
def eval_chebyt(n, x, a = 0, b = 0):
    if a == b:
        norm = 1
    else:
        if n == 0:
            norm = b - a
        elif n == 1:
            norm = (b**2 - a**2) / 2
        elif n == 2:
            norm = (2 * (b**3 - a**3) / 3 - (b - a)) / 2
        elif n == 3:
            norm = (4 * (b**4 - a**4) / 4 - 3 * (b**2 - a**2) / 2) / 4
        elif n == 4:
            norm = (8 * (b**5 - a**5) / 5 - 8 * (b**3 - a**3) / 3 + (b - a)) / 8
        elif n == 5:
            norm = (16 * (b**6 - a**6) / 6 - 20 * (b**4 - a**4) / 4 + 5 * (b**2 - a**2) / 2) / 16

    if n == 0:
        return np.ones_like(x)/norm
    elif n == 1:
        return x/norm
    elif n == 2:
        return (2 * np.power(x, 2) - 1)/norm
    elif n == 3:
        return (4 * np.power(x, 3) - 3 * x)/norm
    elif n == 4:
        return (8 * np.power(x, 4) - 8 * np.power(x, 2) + 1)//norm
    elif n == 5:
        return (16 * np.power(x, 5) - 20 * np.power(x, 3) + 5 * x)//norm

eval_chebyt(1, x)


@njit(fastmath = True)
def gaussian(x, mu, sigma): 
    sigma2 = np.power(sigma, 2)
    return np.power(sigma2*2*pi, -1/2) * np.exp(-np.power(x-mu, 2)/(2*sigma2))
gaussian(x, 1, 2)


@njit(fastmath = True)
def heaviside(x, d):
    return np.where(x >= d, 1.0, 0.0)
heaviside(x, 0)


@njit(fastmath = True)
def nois_log(x):
    return heaviside(x, 0)*np.log(x)
nois_log(x)

@njit(fastmath = True)
def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
factorial(1)


@njit(fastmath = True)
def factoriall(n):
    result = np.zeros_like(n)
    for i in range(n.size):
        result[i] = factorial(n[i])
    return result
factoriall(x)


@njit(fastmath = True)
def puasson(x, n):
    return np.exp(-x)*np.power(x, n)*np.power(factoriall(n), -1)
puasson(x, n)



@njit(fastmath=True)
def exp_dis(x, lam, a=0, b=0):
    if a == b:
        normalization_factor = 1
    else:
        normalization_factor = lam / (np.exp(lam * (b)) - np.exp(lam * (a)))
    return normalization_factor * np.exp(lam * x)


def normalization(counts, bin_edges):
    total_counts = np.sum(counts)
    bin_width = bin_edges[1] - bin_edges[0]
    return bin_width * total_counts

from iminuit import Minuit

def max_bin_lik(f, bin_centers, counts, args0, bounds=None, err_need = False):
    normalization = np.max(counts)//10
    counts = counts/normalization
    norm = np.sum(counts)
    args0["norm"] = norm

    pdf = lambda x, **args: f(x, **{k: v for k, v in args.items() if k != "norm"}) * args["norm"]

    if bounds != None:
        bounds["norm"] = (0, norm*100)
    def df(*args):
        current_args = {k: v for k, v in zip(args0.keys(), args)}
        return -np.sum(np.log(puasson(pdf(bin_centers, **current_args), counts)))
    
    minuit = Minuit(df, *[args0[__key] for __key in args0.keys()], name=args0.keys())
    minuit.migrad()
    rez = minuit.values.to_dict()
    print(rez)
    rez["norm"] = rez["norm"]* normalization
    if err_need:
        return rez, pdf, norm, minuit.errors
    return rez, pdf


def max_lik(f, x, args0, bounds=None, err_need = False):

    pdf = lambda x, **args: f(x, **{k: v for k, v in args.items() if k != "norm"}) * args["norm"]

    def df(*args):
        current_args = {k: v for k, v in zip(args0.keys(), args)}
        return -np.sum(np.log(pdf(x, **current_args)))
    
    minuit = Minuit(df, *[args0[__key] for __key in args0.keys()], name=args0.keys())
    minuit.migrad()
    rez = minuit.values.to_dict()
    print(rez)
    if err_need:
        return rez, pdf, minuit.errors
    return rez, pdf


def errorhist(data, bins=10, fmt='o', err_func = np.sqrt, axs = plt, density=False, **kwargs):
        counts, bin_edges = np.histogram(data, bins=bins)
        if density:
            errors = err_func(counts) / np.sum(counts * np.diff(bins))
            counts = counts / np.sum(counts * np.diff(bins)) 
        else:
            errors = err_func(counts)
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        axs.errorbar(bin_centers, counts, yerr=errors, fmt=fmt, **kwargs)
        return counts, bin_centers


from typing import List, Tuple, Optional, Union
import pandas as pd

def compute_histogram(
    dataset: ds.dataset,
    bins: np.ndarray,
    target: Union[str, list],
    fun=lambda x: x,
    filter_mask: Optional[pc.Expression] = None,
    norm: bool = False,
    nanto: float = np.nan
) -> Tuple[np.ndarray, np.ndarray, int]:

    scanner = dataset.scanner(batch_size=100_000, filter=filter_mask)
    hist_counts = np.zeros(len(bins) - 1)

    for batch in scanner.to_batches():
        table = Table.from_batches([batch])

        if isinstance(target, (list, tuple)):
            df = table.select(target).to_pandas()
            df = df.fillna(nanto)
            values = fun(df.values.T)  # shape (N, len(target))
        else:
            df = table.select([target]).to_pandas()
            values = fun(df[target].fillna(nanto).values)

        counts, _ = np.histogram(values, bins=bins)
        hist_counts += counts

    total_events = int(np.sum(hist_counts))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    if norm and total_events > 0:
        hist_counts /= total_events

    return bin_centers, hist_counts, total_events    


def compute_nd_histogram(
    dataset: ds.Dataset,
    bins: Union[np.ndarray, List[np.ndarray]],
    targets: Union[str, List[str]],
    fun=lambda x: x,
    filter_mask: Optional[pc.Expression] = None,
    norm: bool = False,
    nanto: float = np.nan
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    if isinstance(targets, str):
        targets = [targets]

    if isinstance(bins, np.ndarray):
        bins = [bins] * len(targets)

    bin_centers = [0.5 * (b[:-1] + b[1:]) for b in bins]
    hist_counts = np.zeros_like(np.meshgrid(*bin_centers, indexing='ij')[0])
    scanner = dataset.scanner(batch_size=100_000, filter=filter_mask)

    for batch in scanner.to_batches():
        table = Table.from_batches([batch])
        df = table.select(targets).to_pandas()
        df = df.fillna(nanto)
        values = df.values

        counts, _ = np.histogramdd(values, bins=bins)
        hist_counts += counts

    total_events = int(np.sum(hist_counts))

    # Вычисляем центры биннов по каждой оси

    if norm and total_events > 0:
        hist_counts /= total_events
    bin_centers = np.meshgrid(*bin_centers, indexing='ij')
    return bin_centers, hist_counts, total_events


def get_values(dataset, target: List[str], filter_mask = None, max_size_gb: float = 3.0) -> Tuple[pd.DataFrame]:
    scanner = dataset.scanner(batch_size=100_000, filter=filter_mask)
    parts = []
    total_bytes = 0
    for batch in scanner.to_batches():
        table = Table.from_batches([batch])
        df_part = table.select(target).to_pandas()
        del table
        part_bytes = df_part.memory_usage(deep=True).sum()
        total_bytes += part_bytes

        if total_bytes > max_size_gb * 1e9:
            del scanner, batch, df_part, parts
            raise MemoryError(f"Превышен лимит данных: {total_bytes / 1e9:.2f} GB > {max_size_gb} GB")

        parts.append(df_part)
        del df_part
    if total_bytes < 1e9:
        print(f"Total data size: {total_bytes / 1e6:.2f} MB")
    else:
        print(f"Total data size: {total_bytes / 1e9:.2f} GB")
    del scanner, batch
    dt = pd.concat(parts, ignore_index=True)
    del parts, part_bytes, total_bytes
    return dt



class hist_approx:
    def __init__(self) -> None:
        self.kdtree: Union[None, 'KDTree'] = None
        self.points: Union[None, np.ndarray] = None
        self.N: Union[None, float] = None
        self.hist_counts: Union[None, np.ndarray] = None
        self.bin_centers: Union[None, List[np.ndarray]] = None

    def put_hist(self, bin_centers: List[np.ndarray], hist_counts: np.ndarray) -> None:
        from scipy.spatial import KDTree

        self.N = np.sum(hist_counts)
        bin_widths = [np.mean(np.diff(np.unique(bc))) for bc in bin_centers]
        self.bin_volume = np.prod(bin_widths)  # V_bin
        self.hist_counts = hist_counts / self.N / self.bin_volume
        self.bin_centers = bin_centers

        self.points = np.stack([m.reshape(-1) for m in bin_centers], axis=-1)
        self.kdtree = KDTree(self.points)

    def get_pdf(self, x: List[np.ndarray]) -> np.ndarray:
        sp = x[0].shape
        x_flat = np.stack([np.asarray(xi).ravel() for xi in x], axis=-1)
        _, idxs = self.kdtree.query(x_flat)
        result = self.hist_counts.ravel()[idxs]
        return result.reshape(sp)

    def get_counts(self, x: List[np.ndarray]) -> np.ndarray:
        sp = x[0].shape
        x_flat = np.stack([np.asarray(xi).ravel() for xi in x], axis=-1)
        _, idxs = self.kdtree.query(x_flat)
        result = self.hist_counts.ravel()[idxs] * self.N * self.bin_volume
        return result.reshape(sp)

    def save_hist(self, filename: str) -> None:
        import json
        data_to_save = {
            "bin_centers": [bc.tolist() for bc in self.bin_centers],
            "counts": (self.hist_counts * self.N).tolist(),
            "N": self.N
        }
        with open(filename, "w") as f:
            json.dump(data_to_save, f, indent=4)

    @classmethod
    def load_hist(cls, filename: str) -> 'hist_approx':
        import json

        with open(filename, "r") as f:
            data = json.load(f)

        bin_centers = [np.array(bc) for bc in data["bin_centers"]]
        hist_counts = np.array(data["counts"])
        hist_counts = hist_counts.reshape(bin_centers[0].shape)

        obj = cls()
        obj.put_hist(bin_centers, hist_counts)
        return obj
