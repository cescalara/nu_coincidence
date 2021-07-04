---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.0
  kernelspec:
    display_name: cosmic_coincidence
    language: python
    name: cosmic_coincidence
---

```python
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import ligo.skymap.plot
from astropy import units as u
```

```python
import sys
sys.path.append("../")
```

## Using cosmic_coincidence

```python
from cosmic_coincidence.coincidence.blazar_nu import BlazarNuConnection
from cosmic_coincidence.popsynth_wrapper import PopsynthParams, PopsynthWrapper
from cosmic_coincidence.neutrinos.icecube import IceCubeObsParams, IceCubeTracksWrapper
from cosmic_coincidence.neutrinos.icecube import IceCubeAlertsParams, IceCubeAlertsWrapper
from cosmic_coincidence.utils.package_data import get_path_to_data
from cosmic_coincidence.utils.plotting import SphericalCircle
```

```python
seed = 43
```

```python
bllac_spec = get_path_to_data("bllac_connected.yml")
bllac_param_server = PopsynthParams(bllac_spec)
bllac_param_server.seed = seed
#aux = bllac_param_server.pop_spec["auxiliary samplers"]
#aux["flare_rate"]["index"] = 1.9
bllac_pop = PopsynthWrapper(bllac_param_server)


fsrq_spec = get_path_to_data("fsrq_connected.yml")
fsrq_param_server = PopsynthParams(fsrq_spec)
fsrq_param_server.seed = seed
fsrq_pop = PopsynthWrapper(fsrq_param_server)
```

```python
#nu_spec = "output/connected_tracks.yml"
#nu_param_server = IceCubeObsParams.from_file(nu_spec)
#nu_param_server.seed = 42
#nu_obs = IceCubeTracksWrapper(nu_param_server)
```

```python
hese_nu_spec = "output/connected_hese_nu.yml"
ehe_nu_spec = "output/connected_ehe_nu.yml"
nu_param_server = IceCubeAlertsParams(hese_nu_spec, ehe_nu_spec)
nu_param_server.seed = seed
flux_factor = 0.0005
nu_param_server.hese.connection["flux_factor"] = flux_factor
nu_param_server.ehe.connection["flux_factor"] = flux_factor
nu_obs = IceCubeAlertsWrapper(nu_param_server)
```

```python
blazar_nu = BlazarNuConnection(bllac_pop, fsrq_pop, nu_obs, flare_only=False)
```

```python
bc = blazar_nu.bllac_connection
fc = blazar_nu.fsrq_connection

print("BL Lac nu:", len(bc["nu_ras"]))
print("FSRQ nu:", len(fc["nu_ras"]))
```

```python
colors = plt.cm.viridis(np.linspace(0, 1, 10))
plt.style.use("minimalist")
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
np.random.seed(42)
factor = 0.1
fs = 20
# BL Lacs
max_nex = max(bc["Nnu_ex_steady"]*factor)
for ra, dec, nex in zip(bllac_pop.survey.ra, bllac_pop.survey.dec, 
                   bc["Nnu_ex_steady"]*factor):
    n = np.random.poisson(nex)
    if n > 0:
        if nex/max_nex < 0.1:
            alpha = 0.1
        else:
            alpha = nex/max_nex
            ax.text(ra+3, dec+3, "%.2f" % nex, transform=ax.get_transform("icrs"), 
                    fontsize=fs)
        circle = SphericalCircle((ra * u.deg, dec * u.deg), 1.5 * u.deg,
                                 transform=ax.get_transform("icrs"), alpha=alpha, 
                                 color=colors[0])
        ax.add_patch(circle)
    
# FSRQs
max_nex = max(fc["Nnu_ex_steady"]*factor)
for ra, dec, nex in zip(fsrq_pop.survey.ra, fsrq_pop.survey.dec, 
                   fc["Nnu_ex_steady"]*factor):
    n = np.random.poisson(nex)
    if n > 0:
        if nex/max_nex < 0.1:
            alpha = 0.1
        else:
            alpha = nex/max_nex
            ax.text(ra+3, dec+3, "%.2f" % nex, transform=ax.get_transform("icrs"), 
                    fontsize=fs)
        circle = SphericalCircle((ra * u.deg, dec * u.deg), 1.5 * u.deg,
                                 transform=ax.get_transform("icrs"), alpha=alpha, 
                                 color=colors[0])
        ax.add_patch(circle)
ax.axis("off")
fig.savefig("figures/sky_template_4.pdf", bbox_inches="tight", dpi=200)
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
for r, d, e, det in zip(np.rad2deg(bc["nu_ras"]), np.rad2deg(bc["nu_decs"]), 
                   bc["nu_ang_errs"], bc["src_detected"]):
    if det:
        color = "green"
    else:
        color = "red"
    circle = SphericalCircle((r * u.deg, d * u.deg), e * 2 * u.deg,
                             transform=ax.get_transform("icrs"), alpha=0.7, 
                             color=color)
    ax.add_patch(circle)
```

```python
# select only contribution from flares
sel = bc["src_flare"] == 1
sum(bc["src_flare"])
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
for r, d, e, det in zip(np.rad2deg(bc["nu_ras"][sel]), np.rad2deg(bc["nu_decs"][sel]), 
                   bc["nu_ang_errs"][sel], bc["src_detected"][sel]):
    if det:
        color = "green"
    else:
        color = "red"
    circle = SphericalCircle((r * u.deg, d * u.deg), e * 2 * u.deg,
                             transform=ax.get_transform("icrs"), alpha=0.7, 
                             color=color)
    ax.add_patch(circle)
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
for r, d, e, det in zip(np.rad2deg(fc["nu_ras"]), np.rad2deg(fc["nu_decs"]), 
                   fc["nu_ang_errs"], fc["src_detected"]):
    if det:
        color = "green"
    else:
        color = "red"
    circle = SphericalCircle((r * u.deg, d * u.deg), e * 2 * u.deg,
                             transform=ax.get_transform("icrs"), alpha=0.7, 
                             color=color)
    ax.add_patch(circle)
```

```python
# select only contribution from flares
sel = fc["src_flare"] == 1
sum(fc["src_flare"])
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
for r, d, e, det in zip(np.rad2deg(fc["nu_ras"][sel]), np.rad2deg(fc["nu_decs"][sel]), 
                   fc["nu_ang_errs"][sel], fc["src_detected"][sel]):
    if det:
        color = "green"
    else:
        color = "red"
    circle = SphericalCircle((r * u.deg, d * u.deg), e * 2 * u.deg,
                             transform=ax.get_transform("icrs"), alpha=0.7, 
                             color=color)
    ax.add_patch(circle)
```

```python
redshifts = []
for i in bc["src_id"]:
    z = blazar_nu._bllac_pop.survey.distances[i]
    redshifts.append(z)
    
max(redshifts)
```

```python
redshifts = []
for i in fc["src_id"]:
    z = blazar_nu._fsrq_pop.survey.distances[i]
    redshifts.append(z)
    
max(redshifts)
```

```python
sum(fc["src_flare"])
```

### Testing constraints

```python
import h5py
from joblib import Parallel, delayed
```

```python
from cosmic_coincidence.coincidence.blazar_nu import BlazarNuConnection
from cosmic_coincidence.popsynth_wrapper import PopsynthParams, PopsynthWrapper
from cosmic_coincidence.neutrinos.icecube import IceCubeObsParams, IceCubeTracksWrapper
from cosmic_coincidence.neutrinos.icecube import IceCubeAlertsParams, IceCubeAlertsWrapper
from cosmic_coincidence.utils.package_data import get_path_to_data
from cosmic_coincidence.utils.plotting import SphericalCircle
```

```python
def run_sims(ntrials, i):

    output = {}
    
    output["flux_factor"] = []
    output["f_var_j_bl"] = []
    output["f_duty_j_bl"] = []
    output["f_var_j_fs"] = []
    output["f_duty_j_fs"] = []
    
    output["n_alerts_j_bl"] = []
    output["n_alerts_jflare_bl"] = []
    output["n_alerts_j_fs"] = []
    output["n_alerts_jflare_fs"] = []
    
    output["n_multi_j_bl"] = []
    output["n_multi_jflare_bl"] = []
    output["n_multi_j_fs"] = []
    output["n_multi_jflare_fs"] = []
    
    for j in range(ntrials):
        
        seed = 100 * j + i

        bllac_spec = get_path_to_data("bllac_connected.yml")
        bllac_param_server = PopsynthParams(bllac_spec)
        bllac_param_server.seed = seed
        aux = bllac_param_server.pop_spec["auxiliary samplers"]
        aux["variability"]["weight"] = np.random.uniform(0.02, 0.12)
        aux["flare_rate"]["index"] = np.random.uniform(1.5, 2.5)
        aux["flare_durations"]["index"] = np.random.uniform(1.5, 2.5)
        bllac_pop = PopsynthWrapper(bllac_param_server)

        fsrq_spec = get_path_to_data("fsrq_connected.yml")
        fsrq_param_server = PopsynthParams(fsrq_spec)
        fsrq_param_server.seed = seed
        aux = fsrq_param_server.pop_spec["auxiliary samplers"]
        aux["variability"]["weight"] = np.random.uniform(0.35, 0.45)
        aux["flare_rate"]["index"] = np.random.uniform(1.5, 2.5)
        aux["flare_durations"]["index"] = np.random.uniform(1.5, 2.5)
        fsrq_pop = PopsynthWrapper(fsrq_param_server)

        flux_factor = 10**np.random.uniform(-3, -1)
        output["flux_factor"].append(flux_factor)
        hese_nu_spec = "output/connected_hese_nu.yml"
        ehe_nu_spec = "output/connected_ehe_nu.yml"
        nu_param_server = IceCubeAlertsParams(hese_nu_spec, ehe_nu_spec)
        nu_param_server.seed = seed
        nu_param_server.hese.connection["flux_factor"] = flux_factor
        nu_param_server.ehe.connection["flux_factor"] = flux_factor
        nu_obs = IceCubeAlertsWrapper(nu_param_server)

        # Effective flare efficiency
        output["f_var_j_bl"].append(sum(bllac_pop.survey.variability) / 
                                  len(bllac_pop.survey.variability))
        dur_var = [sum(fd) for fd in bllac_pop.survey.flare_durations if fd.size>0]
        output["f_duty_j_bl"].append(np.mean(np.array(dur_var) / 
                bllac_pop.survey.truth["flare_times"]["obs_time"]))
        output["f_var_j_fs"].append(sum(fsrq_pop.survey.variability) / 
                                  len(fsrq_pop.survey.variability))
        dur_var = [sum(fd) for fd in fsrq_pop.survey.flare_durations if fd.size>0]
        output["f_duty_j_fs"].append(np.mean(np.array(dur_var) / 
                fsrq_pop.survey.truth["flare_times"]["obs_time"]))
        
        blazar_nu = BlazarNuConnection(bllac_pop, fsrq_pop, nu_obs)

        bc = blazar_nu.bllac_connection
        fc = blazar_nu.fsrq_connection
        #det = bc["src_detected"].astype(bool)
        flare_bl = bc["src_flare"].astype(bool)
        flare_fs = fc["src_flare"].astype(bool)

        output["n_alerts_j_bl"].append(len(bc["nu_ras"]))
        output["n_alerts_jflare_bl"].append(len(bc["nu_ras"][flare_bl]))
        output["n_alerts_j_fs"].append(len(fc["nu_ras"]))
        output["n_alerts_jflare_fs"].append(len(fc["nu_ras"][flare_fs]))
                       
        unique, counts = np.unique(bc["src_id"], return_counts=True)
        output["n_multi_j_bl"].append(len(counts[counts>1]))
        unique, counts = np.unique(bc["src_id"][flare_bl], return_counts=True)
        output["n_multi_jflare_bl"].append(len(counts[counts>1]))
        unique, counts = np.unique(fc["src_id"], return_counts=True)
        output["n_multi_j_fs"].append(len(counts[counts>1]))
        unique, counts = np.unique(fc["src_id"][flare_fs], return_counts=True)
        output["n_multi_jflare_fs"].append(len(counts[counts>1]))
        
    return output
```

```python
#flux_factors = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]
flux_factors = [1e-4, 1e-2]
ntrials = 2

flux_factor = []
f_var_bl = []
f_var_fs = []
f_duty_bl = []
f_duty_fs = []

n_alerts_tot_bl = []
n_multi_tot_bl = []
n_alerts_flare_bl = []
n_multi_flare_bl = []

n_alerts_tot_fs = []
n_multi_tot_fs = []
n_alerts_flare_fs = []
n_multi_flare_fs = []
   
out = Parallel(n_jobs=4)(delayed(run_sims)(ntrials, i) for i in range(4))

for output in out:
    
    flux_factor.append(output["flux_factor"])
    
    f_var_bl.append(output["f_var_j_bl"])
    f_var_fs.append(output["f_var_j_fs"])
    f_duty_bl.append(output["f_duty_j_bl"])
    f_duty_fs.append(output["f_duty_j_fs"])
    
    n_alerts_tot_bl.append(output["n_alerts_j_bl"])
    n_alerts_tot_fs.append(output["n_alerts_j_fs"])
    n_alerts_flare_bl.append(output["n_alerts_jflare_bl"])
    n_alerts_flare_fs.append(output["n_alerts_jflare_fs"])
    
    n_multi_tot_bl.append(output["n_multi_j_bl"])
    n_multi_tot_fs.append(output["n_multi_j_fs"])
    n_multi_flare_bl.append(output["n_multi_jflare_bl"])
    n_multi_flare_fs.append(output["n_multi_jflare_fs"])
```

```python
alert_th = 50
multi_th = 0
```

```python
ff = np.array(flux_factor).flatten()
fvar = np.array(f_var_bl).flatten()
alerts = np.array(n_alerts_tot_bl).flatten()
```

```python
alerts
```

```python
fig, ax = plt.subplots()
sel = alerts < 20
ax.scatter(ff, fvar)
ax.scatter(ff[sel], fvar[sel])
```

```python
fig, ax = plt.subplots()
ax.plot(flux_factors, [len(np.array(n_a)[np.array(n_a) > alert_th]) 
                       for n_a in n_alerts_tot_bl], 
        label="BL Lac tot")
ax.plot(flux_factors, [len(np.array(n_a)[np.array(n_a) > alert_th]) 
                       for n_a in n_alerts_flare_bl], 
        label="BL Lac flare")
ax.plot(flux_factors, [len(np.array(n_a)[np.array(n_a) > alert_th]) 
                       for n_a in n_alerts_tot_fs], 
        label="FSRQ tot")
ax.plot(flux_factors, [len(np.array(n_a)[np.array(n_a) > alert_th]) 
                       for n_a in n_alerts_flare_fs], 
        label="FSRQ flare")
ax.set_xscale("log")
ax.legend()
```

```python
fig, ax = plt.subplots()
ax.plot(flux_factors, [len(np.array(n_m)[np.array(n_m) > multi_th]) 
                       for n_m in n_multi_tot_bl], 
        label="BL Lac tot")
ax.plot(flux_factors, [len(np.array(n_m)[np.array(n_m) > multi_th]) 
                       for n_m in n_multi_flare_bl], 
        label="BL Lac flare")
ax.plot(flux_factors, [len(np.array(n_m)[np.array(n_m) > multi_th]) 
                       for n_m in n_multi_tot_fs], 
        label="FSRQ tot")
ax.plot(flux_factors, [len(np.array(n_m)[np.array(n_m) > multi_th]) 
                       for n_m in n_multi_flare_fs], 
        label="FSRQ flare") 
ax.set_xscale("log")
ax.legend()
```

```python
with h5py.File("output/test_constraints.h5", "w") as f:
    f.create_dataset("flux_factors", data=flux_factors)
    f.create_dataset("ntrials", data=ntrials)
    f.create_dataset("n_alerts_tot_bl", data=n_alerts_tot_bl)
    f.create_dataset("n_alerts_flare_bl", data=n_alerts_flare_bl)
    f.create_dataset("n_alerts_tot_fs", data=n_alerts_tot_fs)
    f.create_dataset("n_alerts_flare_fs", data=n_alerts_flare_fs)
    
    f.create_dataset("n_multi_tot_bl", data=n_multi_tot_bl)
    f.create_dataset("n_multi_flare_bl", data=n_multi_flare_bl)
    f.create_dataset("n_multi_tot_fs", data=n_multi_tot_fs)
    f.create_dataset("n_multi_flare_fs", data=n_multi_flare_fs)
```

```python
with h5py.File("output/test_constraints.h5", "r") as f:
    test = f["n_alerts_tot_bl"][()]
```

```python

```

```python

```

```python

```
