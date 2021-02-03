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
from popsynth.distributions.cosmological_distribution import SFRDistribution
from popsynth.distributions.pareto_distribution import ParetoDistribution
from popsynth.distributions.schechter_distribution import SchechterDistribution
from popsynth.populations import ParetoSFRPopulation
```

```python
# SFR-like evolution
sfr = SFRDistribution()
sfr.r0 = 100
sfr.rise = 2
sfr.decay = 3
sfr.peak = 1

z = np.linspace(0, 5)
fig, ax = plt.subplots()
ax.plot(z, sfr.dNdV(z))
```

```python
# Power law LF
lf = ParetoDistribution()
lf.Lmin = 1e47
lf.alpha = 2

L = 10**np.linspace(47, 50)
fig, ax = plt.subplots()
ax.plot(L, lf.phi(L))
ax.set_xscale("log")
ax.set_yscale("log")
```

```python
# Schecter LF
lf = SchechterDistribution()
lf.Lmin = 5e48
lf.alpha = 0
L = 10**np.linspace(43, 50)

fig, ax = plt.subplots()
ax.plot(L, lf.phi(L))
ax.set_xscale("log")
ax.set_yscale("log")
```

```python

fig, ax = plt.subplots()
ax.hist(lf.draw_luminosity(size=1000))
```

```python
pop_sfr = ParetoSFRPopulation(r0=1, rise=2, decay=3, peak=1, Lmin=1e47, alpha=1, 
                              is_rate=False)
```

```python
my_pop = pop_sfr.draw_survey(boundary=1e2, no_selection=True)
```

```python
_ = my_pop.display_distances()
```

```python
_ = my_pop.display_fluxes()
```

```python
#_ = my_pop.display_flux_sphere()
```

```python

```
