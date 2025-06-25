import numpy as np
from scipy.optimize import curve_fit, minimize
from numpy import log, sqrt, exp, pi, e
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import os
from numba import njit
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL

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
#plt.rcParams['figure.constrained_layout.use'] = True
#plt.rcParams['text.usetex'] = True

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

def rm(fname):
    if os.path.exists(fname):
        os.remove(fname)
    else:
        print("The file does not exist") 

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


def max_lik(f, x, args0, cross = None, bounds = None, fix = None):
    if cross:
        param_ind = x[cross[0]] - 1
        crosss_params = cross[1:]
        l = len(cross[1:])
        def df(*args0):
            args0 = np.array(args0)
            i = 0
            lik_val = 0
            while i <= len(args0):
                if i in crosss_params:
                    lik_val += np.sum(np.log(f(*[_[param_ind == i] for _ in x], args0[i], *args0[l:])))
                i += 1
            return -2*lik_val
    else:
        print(-np.sum(2*np.log(f(x, *args0))))
        def df(*args0):
            return -np.sum(2*np.log(f(x, *args0)))
    print(df(*args0))
    minuit = Minuit(df, *args0)
    if None != bounds:
        minuit.limits = bounds
    if not (fix is None):
        minuit.fixed()
    minuit.migrad()
    rez = minuit.values
    print(minuit.values)
    print(minuit.errors)
    print(df(*minuit.values))
    return minuit.values, minuit.errors

#
def errorhist(data, bins=10, fmt='.', color='dimgrey', err_func = np.sqrt, axs = plt, markersize=10, label = None):
        counts, bin_edges = np.histogram(data, bins=bins)
        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
        axs.errorbar(bin_centers, counts, yerr=err_func(counts), fmt=fmt, color=color, markersize=markersize, label = label)
        return counts, bin_centers


from typing import List, Tuple, Optional
import pandas as pd
import pyarrow as pa

def compute_histogram(dataset, bins, target, fun = lambda x: x, filter_mask = None, norm = False) -> Tuple[np.ndarray[np.float64], np.ndarray[np.int64], int]:
    scanner = dataset.scanner(batch_size=100_000, filter=filter_mask)
    hist_counts = np.zeros(len(bins) - 1)
    total_events = 0

    for batch in scanner.to_batches():
        table = pa.Table.from_batches([batch])
        dt = table[target].to_numpy()
        counts, _ = np.histogram(fun(dt), bins=bins)
        hist_counts += counts
        total_events += len(dt)
        del dt, counts, _, table
    del scanner, batch

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    if norm:
        hist_counts /= np.sum(hist_counts)
    return bin_centers, hist_counts, total_events

def get_values(dataset, target: List[str], filter_mask = None, max_size_gb: float = 3.0) -> Tuple[pd.DataFrame]:
    scanner = dataset.scanner(batch_size=100_000, filter=filter_mask)
    parts = []
    total_bytes = 0
    for batch in scanner.to_batches():
        table = pa.Table.from_batches([batch])
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


#class CustomedAxes(Axes):
#    def errorhist(self, data, bins=10, fmt='.', color='dimgrey', err_func = np.sqrt, **kwargs):
#        counts, bin_edges = np.histogram(data, bins=bins, **kwargs)
#        bin_centers = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
#        self.errorbar(bin_centers, counts, yerr=err_func(counts), fmt=fmt, color=color, **kwargs)
#        return counts, bin_centers
#
#__subplot = plt.subplots
#
#def subplots(nrows=1, ncols=1, **kwargs):
#    fig, axes = __subplot(nrows=nrows, ncols=ncols, layout='constrained', **kwargs)
#    if isinstance(axes, np.ndarray):
#        custom_axes = []
#        for ax in axes.flat:
#            position = ax.get_position()
#            ax.remove() 
#            custom_ax = CustomedAxes(fig, position)
#            fig.add_axes(custom_ax)
#            custom_axes.append(custom_ax)
#        custom_axes = np.array(custom_axes).reshape(axes.shape)
#    else:
#        position = axes.get_position()
#        axes.remove()
#        custom_axes = CustomedAxes(fig, position)
#        fig.add_axes(custom_axes)
#    return fig, custom_axes
#
#plt.subplots = subplots


