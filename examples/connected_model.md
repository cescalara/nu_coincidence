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
seed = 100
```

```python
bllac_spec = get_path_to_data("bllac_connected.yml")
bllac_param_server = PopsynthParams(bllac_spec)
bllac_param_server.seed = seed
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
nu_obs = IceCubeAlertsWrapper(nu_param_server)
```

```python
blazar_nu = BlazarNuConnection(bllac_pop, fsrq_pop, nu_obs)
```

```python
bc = blazar_nu.bllac_connection
fc = blazar_nu.fsrq_connection

print("BL Lac nu:", len(bc["nu_ras"]))
print("FSRQ nu:", len(fc["nu_ras"]))
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
flux_factors = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]
ntrials = 10
n_alerts_tot_bl = []
n_multi_tot_bl = []
n_alerts_flare_bl = []
n_multi_flare_bl = []

n_alerts_tot_fs = []
n_multi_tot_fs = []
n_alerts_flare_fs = []
n_multi_flare_fs = []

for i, f in enumerate(flux_factors):
    n_alerts_j_bl = []
    n_multi_j_bl = []
    n_alerts_jflare_bl = []
    n_multi_jflare_bl = []
    
    n_alerts_j_fs = []
    n_multi_j_fs = []
    n_alerts_jflare_fs = []
    n_multi_jflare_fs = []
    
    for j in range(ntrials):
        
        seed = 100 * j + i

        bllac_spec = get_path_to_data("bllac_connected.yml")
        bllac_param_server = PopsynthParams(bllac_spec)
        bllac_param_server.seed = seed
        bllac_pop = PopsynthWrapper(bllac_param_server)

        fsrq_spec = get_path_to_data("fsrq_connected.yml")
        fsrq_param_server = PopsynthParams(fsrq_spec)
        fsrq_param_server.seed = seed
        fsrq_pop = PopsynthWrapper(fsrq_param_server)

        hese_nu_spec = "output/connected_hese_nu.yml"
        ehe_nu_spec = "output/connected_ehe_nu.yml"
        nu_param_server = IceCubeAlertsParams(hese_nu_spec, ehe_nu_spec)
        nu_param_server.seed = seed
        nu_param_server.hese.connection["flux_factor"] = f
        nu_param_server.ehe.connection["flux_factor"] = f
        nu_obs = IceCubeAlertsWrapper(nu_param_server)

        blazar_nu = BlazarNuConnection(bllac_pop, fsrq_pop, nu_obs)

        bc = blazar_nu.bllac_connection
        fc = blazar_nu.fsrq_connection
        #det = bc["src_detected"].astype(bool)
        flare_bl = bc["src_flare"].astype(bool)
        flare_fs = fc["src_flare"].astype(bool)

        n_alerts_j_bl.append(len(bc["nu_ras"]))
        n_alerts_jflare_bl.append(len(bc["nu_ras"][flare_bl]))
        n_alerts_j_fs.append(len(fc["nu_ras"]))
        n_alerts_jflare_fs.append(len(fc["nu_ras"][flare_fs]))
                       
        unique, counts = np.unique(bc["src_id"], return_counts=True)
        n_multi_j_bl.append(len(counts[counts>1]))
        unique, counts = np.unique(bc["src_id"][flare_bl], return_counts=True)
        n_multi_jflare_bl.append(len(counts[counts>1]))
        unique, counts = np.unique(fc["src_id"], return_counts=True)
        n_multi_j_fs.append(len(counts[counts>1]))
        unique, counts = np.unique(fc["src_id"][flare_fs], return_counts=True)
        n_multi_jflare_fs.append(len(counts[counts>1]))
        
    
    n_alerts_tot_bl.append(n_alerts_j_bl)
    n_alerts_tot_fs.append(n_alerts_j_fs)
    n_alerts_flare_bl.append(n_alerts_jflare_bl)
    n_alerts_flare_fs.append(n_alerts_jflare_fs)
    
    n_multi_tot_bl.append(n_multi_j_bl)
    n_multi_tot_fs.append(n_multi_j_fs)
    n_multi_flare_bl.append(n_multi_jflare_bl)
    n_multi_flare_fs.append(n_multi_jflare_fs)
```

```python
alert_th = 50
multi_th = 0
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

```
