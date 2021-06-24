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

```

```python
sum(fc["src_flare"])
```

### Testing constraints

```python
from cosmic_coincidence.coincidence.blazar_nu import BlazarNuConnection
from cosmic_coincidence.popsynth_wrapper import PopsynthParams, PopsynthWrapper
from cosmic_coincidence.neutrinos.icecube import IceCubeObsParams, IceCubeTracksWrapper
from cosmic_coincidence.neutrinos.icecube import IceCubeAlertsParams, IceCubeAlertsWrapper
from cosmic_coincidence.utils.package_data import get_path_to_data
from cosmic_coincidence.utils.plotting import SphericalCircle
```

```python
flux_factors = [0.0001, 0.001, 0.01, 0.1]
n_alerts_tot_bl = []
n_multi_tot_bl = []
n_alerts_tot_fs = []
n_multi_tot_fs = []
```

```python
for i, f in enumerate(flux_factors):
    n_alerts_j_bl = []
    n_multi_j_bl = []
    n_alerts_j_fs = []
    n_multi_j_fs = []
    for j in range(2):
        
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

        n_alerts_j_bl.append(len(bc["nu_ras"]))
        n_alerts_j_fs.append(len(fc["nu_ras"]))
                       
        nu_from_each_bl = bc["Nnu_steady"] + bc["Nnu_flare"]
        multiplet_bl = len(nu_from_each_bl[nu_from_each_bl > 1])
        nu_from_each_fs = fc["Nnu_steady"] + fc["Nnu_flare"]
        multiplet_fs = len(nu_from_each_fs[nu_from_each_fs > 1])
        n_multi_j_bl.append(multiplet_bl)
        n_multi_j_fs.append(multiplet_fs)
    
    n_alerts_tot_bl.append(n_alerts_j_bl)
    n_alerts_tot_fs.append(n_alerts_j_fs)
    
    n_multi_tot_bl.append(n_multi_j_bl)
    n_multi_tot_fs.append(n_multi_j_fs)
```

```python
alert_th = 50
multi_th = 0
```

```python
for n_alert in n_alerts_tot_bl:
    n_alert = np.array(n_alert)
    frac = len(n_alert[n_alert > alert_th])
    print(frac)
    
for n_alert in n_alerts_tot_fs:
    n_alert = np.array(n_alert)
    frac = len(n_alert[n_alert > alert_th])
    print(frac)
```

```python
for n_multi in n_multi_tot_bl:
    n_multi = np.array(n_multi)
    frac = len(n_multi[n_multi > multi_th])
    print(frac)
    
for n_multi in n_multi_tot_fs:
    n_multi = np.array(n_multi)
    frac = len(n_multi[n_multi > multi_th])
    print(frac)
```

```python
len(bc["nu_ras"])
```

```python
sum(bc["Nnu_steady"]) #+ sum(bc["Nnu_flare"])
```

```python

```
