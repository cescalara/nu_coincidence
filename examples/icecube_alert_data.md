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
import pandas as pd
from matplotlib import pyplot as plt
from astropy.time import Time 
```

```python
df = pd.read_csv("data/catalog_of_alerts.txt", delim_whitespace=True, comment="#", 
                names=["MJD", "RA", "RA_Error", "Dec", "Dec_Error"])
```

```python
times = Time(df.MJD.values, format="mjd")
```

```python
fig, ax = plt.subplots()
ax.hist(times.value - times.value[0], bins=4);
plt.vlines(times.value - times.value[0], 0, 2, color='k')
```

```python
print("Events/yr:", len(times) / ((times.value[-1] - times.value[0])/365))
```

```python
print((len(times)-5) / ((times.value[-1] - times.value[0])/365))
```

```python
ra_errs = []
dec_errs = []
Ereco = []
for ra_e, dec_e in zip(df.RA_Error.values, df.Dec_Error.values):
    if ra_e != "(-)" and dec_e != "(-)":
        a, b = eval(ra_e)
        ra_errs.append(a)
        ra_errs.append(b)
        a, b = eval(dec_e)
        dec_errs.append(a)
        dec_errs.append(b)
ra_errs = abs(np.array(ra_errs))
dec_errs = abs(np.array(dec_errs))
```

```python
fig, ax = plt.subplots()
bins=np.linspace(0, 6, 15)
ax.hist(ra_errs, alpha=0.7, label="RA errors", bins=bins)
ax.hist(dec_errs, alpha=0.7, label="Dec errors", bins=bins)
ax.axvline(np.mean(ra_errs), color='k')
ax.axvline(np.mean(dec_errs), color='k')
ax.legend();
```

```python
min(ra_errs)
```

```python
min(dec_errs)
```

```python

import ligo.skymap.plot

import sys
sys.path.append("../")
from cosmic_coincidence.utils.plotting import SphericalCircle
```

```python
ras = df.RA.values
decs = df.Dec.values

fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
ax.scatter(ras, decs, transform=ax.get_transform("icrs"), alpha=0.5)
```

```python
len(decs[decs>0])
```

```python
len(decs[decs<0])
```

## Compare rectangular vs. circular 
Find area of the sky covered by error regions.

```python
event_areas = []
circle_event_areas = []
for ra_e, dec_e in zip(df.RA_Error.values, df.Dec_Error.values):
    if ra_e != "(-)" and dec_e != "(-)":
        ra1, ra2 = eval(ra_e)
        dec1, dec2 = eval(dec_e)

        ra_err = np.deg2rad(abs(ra1) + abs(ra2))
        dec_err = np.deg2rad(abs(dec1) + abs(dec2))
        area = 4 * np.arcsin(np.tan(ra_err/2) * np.tan(dec_err/2))
        event_areas.append(area)
        
        radius_est = np.mean([ra_err, dec_err])/2
        circle_area = 2 * np.pi * (1-np.cos(radius_est))
        circle_event_areas.append(circle_area)
```

```python
sum(event_areas) # steradians
```

```python
sum(circle_event_areas)
```

```python
sum(event_areas) / (4 * np.pi)
```

```python
fig, ax = plt.subplots()
ax.hist(event_areas, alpha=0.5, density=True);
ax.hist(circle_event_areas, alpha=0.5, density=True);
```

```python
min(event_areas)
```

## Simple estimate

```python
N_blazar = 3000
f_var = 0.1
f_flare = 0.1
Omega_nu = 1e-3
R_nu = 7
T_obs = 7.5
```

```python
N_spatial = N_blazar * ((R_nu * T_obs * Omega_nu)/(4*np.pi))
N_spatial
```

```python
N_flare = N_spatial * f_var * f_flare
N_flare
```
