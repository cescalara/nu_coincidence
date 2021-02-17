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
from cosmic_coincidence.utils.interface import Ajello14PDEModel, BLLacLDDEModel
from cosmic_coincidence.utils.interface import FSRQLDDEModel
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

## BL Lac LDDE

```python
ldde = BLLacLDDEModel()
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
popsynth = ldde.popsynth()
```

```python
from cosmic_coincidence.populations.aux_samplers import (VariabilityAuxSampler, 
                                                         FlareRateAuxSampler, 
                                                         FlareTimeAuxSampler,
                                                         FlareDurationAuxSampler)
```

```python
variability = VariabilityAuxSampler()
variability.weight = 0.35

flare_rate = FlareRateAuxSampler()
flare_rate.xmin = 1e-1
flare_rate.index = 1.7

flare_times = FlareTimeAuxSampler()
flare_times.obs_time = 7.5 # years

flare_durations = FlareDurationAuxSampler()

flare_rate.set_secondary_sampler(variability)
flare_times.set_secondary_sampler(flare_rate)
flare_durations.set_secondary_sampler(flare_times)

popsynth.add_observed_quantity(flare_durations)
```

```python
pop = popsynth.draw_survey(boundary=4e-12, hard_cut=True)
#pop = popsynth.draw_survey(boundary=1e2, no_selection=True)
```

```python
N = [len(_) for _ in pop.flare_times]
N_sel = [len(_) for _ in pop.flare_times_selected]
```

```python
fig, ax = plt.subplots()
bins = np.linspace(0, 140)
ax.hist(N);
ax.hist(N_sel)
ax.set_yscale("log")
```

```python
sum(N_sel)
```

```python
n = 0
for i, _ in enumerate(pop.flare_times):
    if pop.flare_times[i] == []:
        n+=1
        #print(pop.flare_times[i])
n/7446
```

```python
d = []
for i, _ in enumerate(pop.flare_durations):
    d.extend(_)
    #print(_)
fig, ax = plt.subplots()
ax.hist(d)
```

```python
list(np.random.uniform(0, [1, 2, 3]))
```

```python
bins = 10**np.linspace(-18, -6)
fig, ax = plt.subplots()
ax.hist(pop.fluxes_latent, bins=bins, alpha=0.5);
ax.hist(pop.selected_fluxes_latent, bins=bins, alpha=0.5)
ax.set_xscale("log")
```

```python
fig, ax = plt.subplots()
ax.scatter(pop.distances, pop.luminosities_latent, alpha=0.1)
ax.scatter(pop.selected_distances, pop.luminosities_latent[pop.selection], alpha=0.1)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim(0.01, 6)
ax.set_ylim(1e44, 1e50)
```

```python
bins=10**np.linspace(44, 52)
fig, ax = plt.subplots()
ax.hist(pop.luminosities_latent[pop.selection], bins=bins);
ax.set_xscale("log")
```

## Testing FSRQs

```python
ldde = FSRQLDDEModel()
ldde.A = 3.06e4
ldde.gamma1 = 0.21
ldde.Lstar = 0.84e48
ldde.gamma2 = 1.58
ldde.zcstar = 1.47
ldde.p1star = 7.35
ldde.tau = 0
ldde.p2 = -6.51
ldde.alpha = 0.21
ldde.mustar = 2.44
ldde.beta = 0
ldde.sigma = 0.18

#ldde.Lmax=1e50
```

```python
from cosmic_coincidence.utils.interface import _zpowerexp, _sfr, _sbpl
from scipy.optimize import curve_fit
```

```python
z = np.linspace(0, ldde.zmax)
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
popsynth = ldde.popsynth()
```

```python
pop = popsynth.draw_survey(boundary=4e-12, hard_cut=True)
```

```python
pop.n_detections
```

```python
bins = 10**np.linspace(-18, -6)
fig, ax = plt.subplots()
ax.hist(pop.fluxes_latent, bins=bins, alpha=0.5);
ax.hist(pop.selected_fluxes_latent, bins=bins, alpha=0.5)
ax.set_xscale("log")
```

```python
fig, ax = plt.subplots()
ax.scatter(pop.distances, pop.luminosities_latent, alpha=0.3)
ax.scatter(pop.selected_distances, pop.luminosities_latent[pop.selection], alpha=0.3)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim(0.01, 10)
ax.set_ylim(1e44, 1e50)
```

```python
bins=10**np.linspace(44, 52)
fig, ax = plt.subplots()
ax.hist(pop.luminosities_latent[pop.selection], bins=bins);
ax.set_xscale("log")
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
