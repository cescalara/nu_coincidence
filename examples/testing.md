---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: cosmic_coincidence
    language: python
    name: cosmic_coincidence
---

# Testing

```python
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append("../")
from cosmic_coincidence.populations.blazars import ToyBLLacPopulation
from cosmic_coincidence.distributions.sbpl_distribution import SBPLDistribution, sbpl

from popsynth.distributions.cosmological_distribution import SFRDistribution
from popsynth.distributions.bpl_distribution import BPLDistribution
from popsynth.populations.pareto_populations import ParetoSFRPopulation
```

```python
# SBPL LF
lf = SBPLDistribution()
lf.Lmin = 1e43
lf.Lbreak = 1e47
lf.Lmax = 1e49
lf.alpha = 1.5
lf.beta = 2.7

L = 10**np.linspace(np.log10(lf.Lmin), np.log10(lf.Lmax))
fig, ax = plt.subplots()
ax.plot(L, lf.phi(L))
ax.hist(lf.draw_luminosity(size=10000), bins=L, density=True)
ax.set_xscale("log")
ax.set_yscale("log")
```

```python
# SFR-like evolution
sfr = SFRDistribution()
sfr.r0 = 10000
sfr.rise = 5
sfr.decay = 2
sfr.peak = 0.1

z = 10**np.linspace(-2, np.log10(1.5))
fig, ax = plt.subplots()
ax.plot(z, sfr.dNdV(z) * sfr.differential_volume(z) / sfr.time_adjustment(z))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("dN/dz")
```

```python
# Broken Power law LF
lf = BPLDistribution()
lf.Lmin = 1e44
lf.Lbreak = 1e47
lf.Lmax = 1e52
lf.alpha = -1.5
lf.beta = -2.1

L = 10**np.linspace(44, 50)
fig, ax = plt.subplots()
ax.plot(L, lf.phi(L))
ax.hist(lf.draw_luminosity(size=10000), bins=L, density=True)
ax.set_xscale("log")
ax.set_yscale("log")
```

```python
pop_bllac = ToyBLLacPopulation(r0=10000, rise=10, decay=3, peak=0.1, Lmin=1e43, 
                               alpha=-1.5, Lbreak=1e47, beta=-2.5, Lmax=1e52, r_max=5)
```

```python
#pop_sfr = ParetoSFRPopulation(r0=10, rise=4, decay=3, peak=1, Lmin=1e43, alpha=1.5)
```

```python
my_pop = pop_bllac.draw_survey(boundary=1e2, no_selection=True)
```

```python
_ = my_pop.display_distances()
```

```python
#_ = my_pop.display_fluxes()
```

```python
#_ = my_pop.display_flux_sphere()
```

```python
fig, ax = plt.subplots()
ax.hist(my_pop.luminosities_latent, bins=10**np.linspace(44, 50))
ax.set_xscale("log")
ax.set_yscale("log")
```
