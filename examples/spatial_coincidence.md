---
jupyter:
  jupytext:
    formats: ipynb,md
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

## Check if point inside circle on a spherical surface

```python
import numpy as np
from matplotlib import pyplot as plt
import ligo.skymap.plot
from astropy import units as u
```

```python
import sys
sys.path.append("../")
from cosmic_coincidence.utils.plotting import SphericalCircle
```

```python
circ_ra = np.deg2rad(130) * u.rad
circ_dec = np.deg2rad(30) * u.rad
circ_alpha = np.deg2rad(20) * u.rad
```

```python
def random_points_on_sphere(N):
    a = np.random.uniform(0, 1, size=N)
    b = np.random.uniform(0, 1, size=N)
    phi = 2 * np.pi * a
    theta = np.arccos((2*b) - 1)
    
    # convert
    ra = phi * u.rad
    dec = (np.pi/2 - theta) * u.rad
    return ra, dec
```

```python
ras, decs = random_points_on_sphere(10000)
```

```python
def get_central_angle(circ_ra, circ_dec, ras, decs):
    sin_term = np.sin(circ_dec) * np.sin(decs) 
    cos_term = np.cos(circ_dec) * np.cos(decs)
    diff = np.cos(circ_ra - ras)
    return np.arccos(sin_term + cos_term * diff)
```

```python
sigmas = get_central_angle(circ_ra.value, circ_dec.value, ras.value, decs.value)
```

```python
selection = sigmas < circ_alpha.value
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
circle = SphericalCircle((circ_ra, circ_dec), circ_alpha, 
                         transform=ax.get_transform("icrs"), alpha=0.5)
ax.add_patch(circle)
ax.scatter(ras.to(u.deg)[~selection], decs.to(u.deg)[~selection], 
           transform=ax.get_transform("icrs"), color='k')
ax.scatter(ras.to(u.deg)[selection], 
           decs.to(u.deg)[selection], 
           transform=ax.get_transform("icrs"), color='r')
```

## Check if point inside ellipse on spherical surface

```python
# this is hard
```
