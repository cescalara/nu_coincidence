---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: cosmic_coincidence
    language: python
    name: cosmic_coincidence
---

```python
import numpy as np
from matplotlib import pyplot as plt
import h5py
from scipy.ndimage.filters import gaussian_filter1d
```

```python
plt.style.use("minimalist")
```

```python
file_name = "output/test_f_constraints_tot_new.h5"
tot = {}
with h5py.File(file_name, "r") as f:
    for key in f:
        tot[key] = f[key][()]
        
file_name = "output/test_f_constraints_flare.h5"
flare = {}
with h5py.File(file_name, "r") as f:
    for key in f:
        flare[key] = f[key][()]

tot["n_alerts_tot"] = tot["n_alerts_tot_bl"] + tot["n_alerts_tot_fs"]
tot["n_alerts_flare"] = tot["n_alerts_flare_bl"] + tot["n_alerts_flare_fs"]
tot["n_multi_tot"] = tot["n_multi_tot_bl"] + tot["n_multi_tot_fs"]
tot["n_multi_flare"] = tot["n_multi_flare_bl"] + tot["n_multi_flare_fs"]

flare["n_alerts_tot"] = flare["n_alerts_tot_bl"] + flare["n_alerts_tot_fs"]
flare["n_alerts_flare"] = flare["n_alerts_flare_bl"] + flare["n_alerts_flare_fs"]
flare["n_multi_tot"] = flare["n_multi_tot_bl"] + flare["n_multi_tot_fs"]
flare["n_multi_flare"] = flare["n_multi_flare_bl"] + flare["n_multi_flare_fs"]
```

## Single event constraints

```python code_folding=[]
# Consistent with 1 alert
alert_th = 1
s = 2

# Steady state + flaring
# All
frac_ok = [len(n_a[(n_a <= alert_th) & (n_a > 0)]) for n_a
           in tot["n_alerts_tot"]] / tot["ntrials"]
frac_ok_smooth = gaussian_filter1d(frac_ok, sigma=s)
# BL Lacs
frac_ok_bl = [len(n_a[(n_a <= alert_th) & (n_a > 0)]) for n_a
           in tot["n_alerts_tot_bl"]] / tot["ntrials"]
frac_ok_bl_smooth = gaussian_filter1d(frac_ok_bl, sigma=s)
# FSRQs
frac_ok_fs = [len(n_a[(n_a <= alert_th) & (n_a > 0)]) for n_a
           in tot["n_alerts_tot_fs"]] / tot["ntrials"]
frac_ok_fs_smooth = gaussian_filter1d(frac_ok_fs, sigma=s)

# Just flaring
# All
frac_ok_flare_0 = [len(n_a[(n_a <= alert_th) & (n_a > 0)]) for n_a 
           in tot["n_alerts_flare"]] / tot["ntrials"] # lower part
frac_ok_flare_1 = [len(n_a[(n_a <= alert_th) & (n_a > 0)]) for n_a
           in flare["n_alerts_tot"]] / flare["ntrials"] # upper part
ffs = np.concatenate([tot["flux_factors"], flare["flux_factors"]])
idx = np.argsort(ffs)
ffs = ffs[idx]
frac_ok_flare = np.concatenate([frac_ok_flare_0, frac_ok_flare_1])[idx]
frac_ok_flare_smooth = gaussian_filter1d(frac_ok_flare[::2], sigma=s)
# BL Lacs
frac_ok_flare_0 = [len(n_a[(n_a <= alert_th) & (n_a > 0)]) for n_a 
           in tot["n_alerts_flare_bl"]] / tot["ntrials"] # lower part
frac_ok_flare_1 = [len(n_a[(n_a <= alert_th) & (n_a > 0)]) for n_a
           in flare["n_alerts_tot_bl"]] / flare["ntrials"] # upper part
ffs = np.concatenate([tot["flux_factors"], flare["flux_factors"]])
idx = np.argsort(ffs)
ffs = ffs[idx]
frac_ok_flare_bl = np.concatenate([frac_ok_flare_0, frac_ok_flare_1])[idx]
frac_ok_flare_bl_smooth = gaussian_filter1d(frac_ok_flare_bl[::2], sigma=s)
# FSRQs
frac_ok_flare_0 = [len(n_a[(n_a <= alert_th) & (n_a > 0)]) for n_a 
           in tot["n_alerts_flare_fs"]] / tot["ntrials"] # lower part
frac_ok_flare_1 = [len(n_a[(n_a <= alert_th) & (n_a > 0)]) for n_a
           in flare["n_alerts_tot_fs"]] / flare["ntrials"] # upper part
ffs = np.concatenate([tot["flux_factors"], flare["flux_factors"]])
idx = np.argsort(ffs)
ffs = ffs[idx]
frac_ok_flare_fs = np.concatenate([frac_ok_flare_0, frac_ok_flare_1])[idx]
frac_ok_flare_fs_smooth = gaussian_filter1d(frac_ok_flare_fs[::2], sigma=s)
```

```python
fig, ax = plt.subplots()
fig.set_size_inches((7,5))
ax.plot(tot["flux_factors"], frac_ok_smooth, label="All emission", lw=3, alpha=0.7, 
        color="blue")
ax.plot(tot["flux_factors"], frac_ok_bl_smooth, lw=2, alpha=0.5, linestyle="--",
        color="blue")
ax.plot(tot["flux_factors"], frac_ok_fs_smooth, lw=2, alpha=0.5, linestyle="-.",
       color="blue")

ax.plot(ffs[::2], frac_ok_flare_smooth, label="Flares only", lw=3, alpha=0.7, 
        color="green")
ax.plot(ffs[::2], frac_ok_flare_bl_smooth, lw=2, alpha=0.5, linestyle="--",
        color="green")
ax.plot(ffs[::2], frac_ok_flare_fs_smooth, lw=2, alpha=0.5, linestyle="-.",
        color="green")
ax.set_xscale("log")
ax.legend(loc=(0.65,0.8))
ax.set_xlabel(r"$\epsilon_{\gamma\nu}$")
ax.set_ylabel(r"Fraction satisfying $N_\nu^a = 1$", labelpad=10)
ax.axhline(0.001, xmin=0.6, xmax=5, lw=3, color="blue", alpha=0.7)
ax.set_ylim(0)
fig.savefig("figures/single_event_const.pdf", bbox_inches="tight", dpi=200)
```

## Total event constraints 

```python
alert_th = 51
multi_th = 0
s = 2
```

```python
# Steady state
# All
frac_ok = [len(n_a[(n_a <= alert_th) & (n_m <= multi_th)]) for n_a, n_m 
            in zip(tot["n_alerts_tot"], tot["n_multi_tot"])] / tot["ntrials"]
frac_ok_smooth = gaussian_filter1d(frac_ok, sigma=s)
# alerts only
frac_ok_a = [len(n_a[(n_a <= alert_th)]) for n_a 
            in tot["n_alerts_tot"]] / tot["ntrials"]
frac_ok_a_smooth = gaussian_filter1d(frac_ok_a, sigma=s)
# multi only
frac_ok_m = [len(n_m[(n_m <= multi_th)]) for n_m 
            in tot["n_multi_tot"]] / tot["ntrials"]
frac_ok_m_smooth = gaussian_filter1d(frac_ok_m, sigma=s)
# BL Lacs
frac_ok_bl = [len(n_a[(n_a <= alert_th) & (n_m <= multi_th)]) for n_a, n_m 
            in zip(tot["n_alerts_tot_bl"], tot["n_multi_tot_bl"])] / tot["ntrials"]
frac_ok_bl_smooth = gaussian_filter1d(frac_ok_bl, sigma=s)
# FSRQs
frac_ok_fs = [len(n_a[(n_a <= alert_th) & (n_m <= multi_th)]) for n_a, n_m 
            in zip(tot["n_alerts_tot_fs"], tot["n_multi_tot_fs"])] / tot["ntrials"]
frac_ok_fs_smooth = gaussian_filter1d(frac_ok_fs, sigma=s)

# Flaring
# All
frac_ok_flare = [len(n_a[(n_a <= alert_th) & (n_m <= multi_th)]) for n_a, n_m 
            in zip(flare["n_alerts_tot"], flare["n_multi_tot"])] / flare["ntrials"]
frac_ok_flare = np.concatenate([[1, 1, 1, 1, 1], frac_ok_flare])
frac_ok_flare_smooth = gaussian_filter1d(frac_ok_flare, sigma=s)
# alerts only
frac_ok_a_flare = [len(n_a[(n_a <= alert_th)]) for n_a 
            in flare["n_alerts_tot"]] / flare["ntrials"]
frac_ok_a_flare_smooth = gaussian_filter1d(frac_ok_a_flare, sigma=s)
# multi only
frac_ok_m_flare = [len(n_m[(n_m <= multi_th)]) for n_m 
            in flare["n_multi_tot"]] / flare["ntrials"]
frac_ok_m_flare_smooth = gaussian_filter1d(frac_ok_m_flare, sigma=s)
# BL Lacs
frac_ok_bl_flare = [len(n_a[(n_a <= alert_th) & (n_m <= multi_th)]) for n_a, n_m 
    in zip(flare["n_alerts_tot_bl"], flare["n_multi_tot_bl"])] / flare["ntrials"]
frac_ok_bl_flare = np.concatenate([frac_ok_bl_flare, [0,0,0,0,0]])
frac_ok_bl_flare_smooth = gaussian_filter1d(frac_ok_bl_flare, sigma=s)

# FSRQs
frac_ok_fs_flare = [len(n_a[(n_a <= alert_th) & (n_m <= multi_th)]) for n_a, n_m 
       in zip(flare["n_alerts_tot_fs"], flare["n_multi_tot_fs"])] / flare["ntrials"]
frac_ok_fs_flare_smooth = gaussian_filter1d(frac_ok_fs_flare, sigma=s)
```

```python
fig, ax = plt.subplots()
fig.set_size_inches((7,5))
ax.plot(tot["flux_factors"], frac_ok_smooth, label="All emission", 
        color="blue", lw=3, alpha=0.7)
ax.plot(tot["flux_factors"], frac_ok_bl_smooth, color="blue", 
        lw=2, alpha=0.5, linestyle="--")
ax.plot(tot["flux_factors"], frac_ok_fs_smooth, color="blue", 
        lw=2, alpha=0.5, linestyle="-.")
ax.plot(flare["flux_factors"], frac_ok_flare_smooth[5:], label="Flares only", 
        color="green", lw=3, alpha=0.7)
ax.plot(np.concatenate([flare["flux_factors"],np.linspace(5, 10, 5)]), 
        frac_ok_bl_flare_smooth, color="green", 
        lw=2, alpha=0.5, linestyle="--")
ax.plot(flare["flux_factors"], 
        frac_ok_fs_flare_smooth, color="green", 
        lw=2, alpha=0.5, linestyle="-.")
ax.axhline(1.0, 0, 0.343, color="green", lw=3, alpha=0.7)
ax.axhline(0.001, xmin=0.6, xmax=5, lw=3, color="blue", alpha=0.7)
ax.set_xscale("log")
ax.legend()
ax.set_ylim(0)
ax.set_xlim(1e-5, 10)
ax.set_xlabel(r"$\epsilon_{\gamma\nu}$")
ax.set_ylabel(r"Fraction satisfying $N_\nu^a \leq 51$ and $N_\nu^m \leq 1$",
             labelpad=10)
fig.savefig("figures/total_event_const.pdf", bbox_inches="tight", dpi=200)
```

## Flare efficiency

```python
from matplotlib.colors import LinearSegmentedColormap
```

```python
file_name = "output/test_eff_flare_constraints.h5"
with h5py.File(file_name, "r") as f:
    flux_factor = f["flux_factor"][()]
    n_bllac = f["n_bllac"][()]
    n_fsrq = f["n_fsrq"][()]
    f_var_bl = f["f_var_bl"][()]
    f_var_fs = f["f_var_fs"][()]
    f_duty_bl = f["f_duty_bl"][()]
    f_duty_fs = f["f_duty_fs"][()]
    ntrials = f["ntrials"][()]
    n_alerts_tot_bl = f["n_alerts_tot_bl"][()]
    n_alerts_flare_bl = f["n_alerts_flare_bl"][()]
    n_alerts_tot_fs = f["n_alerts_tot_fs"][()]
    n_alerts_flare_fs = f["n_alerts_flare_fs"][()]
    n_multi_tot_bl = f["n_multi_tot_bl"][()]
    n_multi_flare_bl = f["n_multi_flare_bl"][()]
    n_multi_tot_fs = f["n_multi_tot_fs"][()]
    n_multi_flare_fs = f["n_multi_flare_fs"][()]
```

```python
alert_th = 51
multi_th = 0
```

```python
# both constraints on whole pop
n_alerts_tot = n_alerts_tot_bl + n_alerts_tot_fs
n_alerts_flare = n_alerts_flare_bl + n_alerts_flare_fs
n_multi_tot = n_multi_tot_bl + n_multi_tot_fs
n_multi_flare = n_multi_flare_bl + n_multi_flare_fs
```

```python
ff_bl = f_var_bl * f_duty_bl
ff_fs = f_var_fs * f_duty_fs
n_tot = n_bllac + n_fsrq
ff = ((n_bllac/n_tot) * ff_bl) + ((n_fsrq/n_tot) * ff_fs)
```

```python
sel = (n_alerts_tot <= alert_th) & (n_multi_tot <= multi_th)
nbins = 10
bins = [np.geomspace(1e-3, 5, nbins), np.linspace(0.002, 0.014, nbins)]
base, _, _ = np.histogram2d(flux_factor, ff, bins=bins)
test, _, _ = np.histogram2d(flux_factor[sel], ff[sel], bins=bins)
hist = np.nan_to_num(test/base).T
```

```python
fig, ax = plt.subplots()
fig.set_size_inches((7,5))
p = ax.pcolormesh(bins[0], bins[1], hist, cmap="Blues")
ax.set_xscale("log")
ax.set_xlim(1e-3, 5)
fig.colorbar(p)
ax.set_ylim()
#ax.set_yscale("log")
```

```python
colors = ["white", "blue"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
```

```python
fig, ax = plt.subplots()
fig.set_size_inches((7,5))
p = ax.contourf(bins[0][:-1], bins[1][:-1], hist, 
                levels=np.linspace(0.1, 1, 10),  cmap="Blues", vmin=0, vmax=1)
p.set_clim(0, 1)
ax.set_xscale("log")
ax.set_xlim(1e-3, 5)
cbar = fig.colorbar(p)
ax.set_xlabel(r"$\epsilon_{\gamma\nu}$")
ax.set_ylabel(r"$\epsilon_\mathrm{flare}$", labelpad=15)
cbar.set_label(r"Fraction satisfying $N_\nu^a \leq 51$ and $N_\nu^m \leq 1$",
               labelpad=10)

fig.savefig("figures/flare_eff_const.pdf", bbox_inches="tight", dpi=200)
```

```python

```

```python

```
