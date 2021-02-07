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
from cosmic_coincidence.populations.sbpl_population import SBPLZPowExpCosmoPopulation
from cosmic_coincidence.distributions.sbpl_distribution import SBPLDistribution
```

```python
# Test pop
pop = SBPLZPowExpCosmoPopulation(r0=92.22, k=11.47, xi=-0.21, 
                                 Lmin=7e43, alpha=-1.32, Lbreak=0.58e48, 
                                 beta=-1.25, Lmax=1e50, r_max=6)
```

```python
z = np.linspace(0, 6)
fig, ax = plt.subplots()
ax.plot(z, pop.spatial_distribution.dNdV(z) * pop.spatial_distribution.differential_volume(z) / pop.spatial_distribution.time_adjustment(z))
```

```python
pop.draw_survey(boundary=1e2, no_selection=True)
```

## Testing Ajello+14 model interface

```python
from scipy import integrate
import sys
sys.path.append("../")
```

```python
from cosmic_coincidence.utils.interface import Ajello14PDEModel
```

```python
pde = Ajello14PDEModel()
pde.A = 78.53 # Mpc^-3 erg^-1 s
pde.Lstar = 0.58e48 # erg s^-1
pde.gamma1 = 1.32
pde.gamma2 = 1.25
pde.kstar = 11.47
pde.xi=-0.21
pde.mustar=2.15
pde.sigma=0.27
```

```python
I1, err = integrate.quad(pde.phi_L, 7e43, 1e52)
print(I1, err)
I1 = I1*1e-13
```

```python
I2, err = integrate.quad(pde.phi_G, 1.45, 2.80)
print(I2, err)
```

```python
print(I1 * I2) # Mpc^-3
```

```python
from astropy import units as u
```

```python
Lambda = I1 * I2 * (1/u.Mpc**3)

Lambda.to(1/u.Gpc**3)
```

## Testing SBPL LF

```python
# SBPL LF
lf = SBPLDistribution()
lf.Lmin = 7e43
lf.Lbreak = 1e48
lf.Lmax = 1e50
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

```
