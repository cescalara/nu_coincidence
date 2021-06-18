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

```python
sum(blazar_nu._bllac_pop.survey.fluxes_observed)
```

```python
sum(bc["src_detected"])/len(bc["nu_ras"])
```

```python
sum(fc["src_detected"])/len(fc["nu_ras"])
```

```python

```
