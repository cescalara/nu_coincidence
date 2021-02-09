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
ax.plot(z, pop.spatial_distribution.dNdV(z) * pop.spatial_distribution.differential_volume(z) / pop.spatial_distribution.time_adjustment(z))
```

```python
pop.draw_survey(boundary=1e2, no_selection=True)
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
ax.set_yscale("log")
```

```python
L = 10**np.linspace(np.log10(ldde.Lmin), np.log10(ldde.Lmax))
fig, ax = plt.subplots()
ax.plot(L, ldde.dNdL(L))
ax.set_xscale("log")
ax.set_yscale("log")
```

```python
# dNdV
L = 10**np.linspace(np.log10(ldde.Lmin), np.log10(ldde.Lmax), 1000)
G = np.linspace(ldde.Gmin, ldde.Gmax, 1000)
zs = np.linspace(ldde.zmin, ldde.zmax)

out = []
for z in zs:
    f = ldde.Phi(L[:,None], z, G) * 1e-13
    out.append(integrate.simps(integrate.simps(f, G), L))
```

```python
def wrap_func(z, A, p):
    return A*np.power(1+z, -p)
popt, pcov = curve_fit(wrap_func, zs, out, p0=(6e-7, 6))
popt
```

```python
fig, ax = plt.subplots()
ax.plot(zs, out)
ax.plot(zs, 6e-7*np.power(1+zs, -6.0))
ax.plot(zs, wrap_func(zs, *popt))
ax.set_yscale("log")
ax.set_ylabel("dN/dV [Mpc^-3]")
ax.set_xlabel("z")
```

```python
# dN/dL
z = np.linspace(ldde.zmin, ldde.zmax, 1000)
G = np.linspace(ldde.Gmin, ldde.Gmax, 1000)
Ls = 10**np.linspace(np.log10(ldde.Lmin), np.log10(ldde.Lmax))

out = []
for L in Ls:
    f = ldde.Phi(L, z[:,None], G) * 1e-13
    out.append(integrate.simps(integrate.simps(f, G), z))
```

```python
from cosmic_coincidence.distributions.sbpl_distribution import sbpl
from scipy.optimize import curve_fit
```

```python
def wrap_func(L, A, Lbreak, a1, a2):
    return A*sbpl(L, ldde.Lmin, Lbreak, ldde.Lmax, a1, a2, limit=True)
p0 = (1, 3e47, 1.6, 2.8)
bounds = ([1e-1, 1e47, 1.0, 2.0], [10, 5e48, 2.0, 3.0])
popt, pcov = curve_fit(wrap_func, Ls, 1e57*np.array(out), p0=p0, bounds=bounds)
popt
```

```python
fig, ax = plt.subplots()
ax.plot(Ls, 1e57*np.array(out))
ax.plot(Ls, sbpl(Ls, ldde.Lmin, 3e47, ldde.Lmax, 1.6, 2.8))
ax.plot(Ls, wrap_func(Ls, *popt))
ax.set_xscale("log")
ax.set_yscale("log")
#ax.set_ylim(1e-5)
```

```python

```
