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
from cosmic_coincidence.utils.interface import Ajello14PDEModel, Ajello14LDDEModel
```

## PDE

```python
pde = Ajello14PDEModel()
pde.A = 78.53 # 1e-13 Mpc^-3 erg^-1 s
pde.Lstar = 0.58e48 # erg s^-1
pde.gamma1 = 1.32
pde.gamma2 = 1.25
pde.kstar = 11.47
pde.xi = -0.21
pde.mustar = 2.15
pde.sigma = 0.27

# For SBPL
pde.Lmax = 1e50
```

```python
pop = pde.popsynth()
```

```python
z = np.linspace(0, 6)
fig, ax = plt.subplots()
ax.plot(z, pop.spatial_distribution.dNdV(z) * pop.spatial_distribution.differential_volume(z) )
```

```python
pde.local_density()
```

```python
# Expects like 2e6 objects!?
#pop.draw_survey(boundary=1e2, no_selection=True)
```

## LDDE

```python
ldde = Ajello14LDDEModel()
ldde.A = 3.39e4
ldde.gamma1 = 0.27
ldde.Lstar = 0.28e48
ldde.gamma2 = 1.86
ldde.zcstar = 1.34
ldde.p1star = 2.24
ldde.tau = 4.92
ldde.p2 = -7.37
ldde.alpha = 4.53e-2
ldde.mustar = 2.1
ldde.beta = 6.46e-2
ldde.sigma = 0.26
```

```python
z = np.linspace(ldde.zmin, ldde.zmax)
fig, ax = plt.subplots()
ax.plot(z, ldde.dNdV(z))
ax.plot(z, ldde.dNdV(z, approx=True))
ax.set_yscale("log")
```

```python
L = 10**np.linspace(np.log10(ldde.Lmin), np.log10(ldde.Lmax))
fig, ax = plt.subplots()
ax.plot(L, ldde.dNdL(L))
ax.plot(L, ldde.dNdL(L, approx=True))
ax.set_xscale("log")
ax.set_yscale("log")
```

```python
# For popsynth
ldde.Lmax = 1e50
pop = ldde.popsynth()
```

```python
pop.draw_survey(boundary=1e2, no_selection=True)
```

## Testing SBPL

```python
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append("../")
from cosmic_coincidence.populations.sbpl_population import SBPLZPowExpCosmoPopulation
from cosmic_coincidence.distributions.sbpl_distribution import SBPLDistribution
```

```python
sbpl = SBPLDistribution(seed=124)
sbpl.Lmin = pop.luminosity_distribution.Lmin
sbpl.Lmax = pop.luminosity_distribution.Lmax
sbpl.Lbreak = pop.luminosity_distribution.Lbreak
sbpl.alpha = pop.luminosity_distribution.alpha
sbpl.beta = pop.luminosity_distribution.beta
```

```python
L = 10**np.linspace(np.log10(pop.luminosity_distribution.Lmin), 
                    np.log10(pop.luminosity_distribution.Lmax))
fig, ax = plt.subplots()
ax.plot(L, sbpl.phi(L))
ax.hist(abs(sbpl.draw_luminosity(10000)), bins=L, density=True)
ax.set_xscale("log")
ax.set_yscale("log")
```

```python
A = sbpl.draw_luminosity(10000)
```

```python
len(A[A<0])
```

```python

```
