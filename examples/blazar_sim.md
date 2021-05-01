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
from scipy import stats
import scipy.special as sf
from matplotlib import pyplot as plt

from popsynth.selection_probability.flux_selectors import HardFluxSelection, SoftFluxSelection

import sys
sys.path.append("../")
from cosmic_coincidence.populations.sbpl_population import SBPLZPowExpCosmoPopulation
from cosmic_coincidence.distributions.sbpl_distribution import SBPLDistribution
from cosmic_coincidence.blazars.fermi_interface import Ajello14PDEModel, _sbpl
from cosmic_coincidence.blazars.bllac import BLLacLDDEModel
from cosmic_coincidence.blazars.fsrq import FSRQLDDEModel
from cosmic_coincidence.populations.aux_samplers import (VariabilityAuxSampler, 
                                                         FlareRateAuxSampler, 
                                                         FlareTimeAuxSampler,
                                                         FlareDurationAuxSampler)
from cosmic_coincidence.populations.selection import GalacticPlaneSelection
```

## General population models

```python
from popsynth.populations.bpl_population import BPLZPowerCosmoPopulation

from cosmic_coincidence.populations.sbpl_population import SBPLZPowerCosmoPopulation
```

### BL Lac

```python
pop_gen = BPLZPowerCosmoPopulation(Lambda=7300, delta=-6, Lmin=7e43, Lmax=1e50, 
                                    alpha=-1.5, Lbreak=1e47, beta=-2.5, r_max=6, 
                                    is_rate=False, seed=42)
```

```python
# Plot distributions
z = np.linspace(0, 6)
fig, ax = plt.subplots()
ax.plot(z, pop_gen.spatial_distribution.dNdV(z) * (1e-9 / (4*np.pi)), color="k") 
ax.set_yscale("log")
ax.set_xlabel("z")
ax.set_ylabel("dNdV [Mpc^-3]")
ax.grid()
ax.set_xlim(0, 3.5)
ax.set_ylim(1e-11, 1e-6)

for i in range(100):
    tmp_gen = BPLZPowerCosmoPopulation(Lambda=np.random.normal(7300, 500), 
                                        delta=np.random.normal(-6, 1), 
                                        Lmin=7e43, Lmax=1e50, 
                                        alpha=-1.5, Lbreak=1e47, beta=-2.5, r_max=6, 
                                        is_rate=False, seed=42)
    ax.plot(z, tmp_gen.spatial_distribution.dNdV(z) * (1e-9 / (4*np.pi)), color="g", 
            alpha=0.1) 
```

```python
L = 10**np.linspace(44, 50)
fig, ax = plt.subplots()
ax.plot(L/1e48, pop_gen.luminosity_distribution.phi(L), color="k")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("L [1e48 erg s^-1]")
ax.set_ylabel("dNdL")
ax.grid()

for i in range(100):
    tmp_gen = BPLZPowerCosmoPopulation(Lambda=7300, 
                                        delta=-6, 
                                        Lmin=7e43, Lmax=1e50, 
                                        alpha=-np.random.normal(1.5, 0.3), 
                                        Lbreak=np.random.normal(1e47, 1e46), 
                                        beta=-np.random.normal(2.5, 0.2), 
                                        r_max=6, 
                                        is_rate=False, seed=42)
    ax.plot(L/1e48, tmp_gen.luminosity_distribution.phi(L), color="g", alpha=0.1) 
```

```python
# Test r_max, Lmin, Lmax, BPL
pop_gen = BPLZPowerCosmoPopulation(Lambda=7300, delta=-6, Lmin=7e43, Lmax=1e52, 
                                    alpha=-1.5, Lbreak=1e47, beta=-2.5, r_max=6, 
                                    is_rate=False, seed=42)
flux_selector = SoftFluxSelection()
flux_selector.boundary = 4e-12
flux_selector.strength = 2

pop_gen.set_flux_selection(flux_selector)

pop = pop_gen.draw_survey(flux_sigma=0.1)
print("Total objects: %i \t Detected objects: %i" % (pop.distances.size, 
                                                    pop.distances[pop.selection].size))
```

```python
# Test number of objects
Ntot = []
Ndet = []
for i in range(100):
    pop_gen = BPLZPowerCosmoPopulation(Lambda=np.random.normal(7300, 500), 
                                       delta=np.random.normal(-6, 1), 
                                       Lmin=7e43, Lmax=1e52, 
                                       alpha=np.random.normal(-1.5, 0.3), 
                                       Lbreak=np.random.normal(1e47, 1e46), 
                                       beta=np.random.normal(-2.5, 0.2), 
                                       r_max=6, 
                                       is_rate=False, 
                                       seed=np.random.randint(100, 10000))
    
    flux_selector = SoftFluxSelection()
    flux_selector.boundary = 4e-12
    flux_selector.strength = 2

    pop_gen.set_flux_selection(flux_selector)

    pop = pop_gen.draw_survey(flux_sigma=0.1)
    Ntot.append(pop.distances.size)
    Ndet.append(pop.distances[pop.selection].size)
```

```python
print("Min Ntot: %i \t Max Ntot: %i" % (min(Ntot), max(Ntot)))
print("Min Ndet: %i \t Max Ndet: %i" % (min(Ndet), max(Ndet)))
```

### FSRQ

```python

```

# Old stuff


## Fermi BL Lac LDDE

```python
# Best fit
ldde = BLLacLDDEModel()
ldde.A = 3.39 * 1e4
ldde.gamma1 = 0.27 
ldde.Lstar = 0.28 * 1e48
ldde.gamma2 = 1.86 
ldde.zcstar = 1.34 
ldde.p1star = 2.24 
ldde.tau = 4.92 
ldde.p2 = -7.37 
ldde.alpha = 4.53 * 1e-2
ldde.mustar = 2.1 
ldde.beta = 6.46 * 1e-2
ldde.sigma = 0.26 
```

```python
# Max 68% CI
ldde = BLLacLDDEModel()
ldde.A = (3.39+7.44) * 1e4
ldde.gamma1 = 0.27 + 0.26
ldde.Lstar = (0.28 + 0.48) * 1e48
ldde.gamma2 = 1.86 + 0.86
ldde.zcstar = 1.34 + 0.22
ldde.p1star = 2.24 + 1.25
ldde.tau = 4.92 + 1.45
ldde.p2 = -7.37 + 2.95
ldde.alpha = (4.53 + 4.98) * 1e-2
ldde.mustar = 2.1 + 0.03
ldde.beta = (6.46 + 2.34) * 1e-2
ldde.sigma = 0.26 + 0.02
```

```python
# Min 68% CI
ldde = BLLacLDDEModel()
ldde.A = (3.39-2.13) * 1e4
ldde.gamma1 = 0.27 - 0.46
ldde.Lstar = (0.28 - 0.21) * 1e48
ldde.gamma2 = 1.86 - 0.48
ldde.zcstar = 1.34 - 0.27
ldde.p1star = 2.24 - 1.07
ldde.tau = 4.92 - 2.12
ldde.p2 = -7.37 - 5.43
ldde.alpha = (4.53 - 6.52) * 1e-2
ldde.mustar = 2.1 - 0.03
ldde.beta = (6.46 - 2.07) * 1e-2
ldde.sigma = 0.26 - 0.02
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
pop_gen = ldde.popsynth()

variability = VariabilityAuxSampler()
variability.weight = 0.05

flare_rate = FlareRateAuxSampler()
flare_rate.xmin = 1/7.5
flare_rate.xmax = 15
flare_rate.index = 1.5

flare_times = FlareTimeAuxSampler()
flare_times.obs_time = 7.5 # years

flare_durations = FlareDurationAuxSampler()

flare_rate.set_secondary_sampler(variability)
flare_times.set_secondary_sampler(flare_rate)
flare_durations.set_secondary_sampler(flare_times)

pop_gen.add_observed_quantity(flare_durations)
```

```python
# Selection
values = 10**np.linspace(-15, -7)
strength = 2
boundary = 4e-12
probs = sf.expit(strength * (np.log10(values) - np.log10(boundary))) 
fig, ax = plt.subplots()
ax.plot(values, probs)
ax.set_xscale("log")
#ax.set_yscale("log")
```

```python
flux_selector = SoftFluxSelection()
flux_selector.boundary = 4e-12
flux_selector.strength = 2

spatial_selector = GalacticPlaneSelection()
spatial_selector.b_limit = 10

pop_gen.set_flux_selection(flux_selector)
pop_gen.add_spatial_selector(spatial_selector)

pop = pop_gen.draw_survey(flux_sigma=0.1)
#pop = pop_gen.draw_survey(boundary=4e-12, hard_cut=True)
#pop = pop_gen.draw_survey(boundary=1e2, no_selection=True)
```

```python
pop.n_detections
```

```python
import ligo.skymap.plot
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
ax.scatter(pop.ra[pop.selection], pop.dec[pop.selection], transform=ax.get_transform("icrs"))
```

```python
len(pop.variability_selected[pop.variability_selected==True]) / len(pop.variability_selected)
```

```python
np.sin(np.deg2rad(5))
```

```python
#pop.display_flux_sphere()
```

```python
len(pop.distances)
```

```python
N_flares = [len(_) for _ in pop.flare_times]
N_flares_det = [len(_) for _ in pop.flare_times_selected]
N_assoc = len([_ for _ in pop.flare_times_selected if _ != []])
fig, ax = plt.subplots()
bins = np.linspace(0, 80)
ax.hist(N_flares, bins=bins, alpha=0.7, label="All")
ax.hist(N_flares_det, bins=bins, alpha=0.7, label="Detected")
ax.set_yscale("log")
ax.set_xlabel("Total number of flares")
ax.legend();
print("N detected flares:", sum(N_flares_det))
print("N associated sources:", N_assoc)
```

```python
d = []
for i, _ in enumerate(pop.flare_durations):
    d.extend(_)
d = np.array(d) * 52 # weeks
bins=np.linspace(min(d), max(d))
fig, ax = plt.subplots()
ax.hist(d, bins=bins, density=True)
ax.plot(bins, stats.pareto(1.5).pdf(bins), alpha=0.7, color='k', 
        label='pareto approx');
ax.set_yscale("log")
ax.set_xlabel("Flare duration (weeks)")
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
pop_gen = ldde.popsynth()

variability = VariabilityAuxSampler()
variability.weight = 0.4

flare_rate = FlareRateAuxSampler()
flare_rate.xmin = 1/7.5
flare_rate.xmax = 15
flare_rate.index = 1.5

flare_times = FlareTimeAuxSampler()
flare_times.obs_time = 7.5 # years

flare_durations = FlareDurationAuxSampler()

flare_rate.set_secondary_sampler(variability)
flare_times.set_secondary_sampler(flare_rate)
flare_durations.set_secondary_sampler(flare_times)

pop_gen.add_observed_quantity(flare_durations)
```

```python
flux_selector = SoftFluxSelection()
flux_selector.boundary = 4e-12
flux_selector.strength = 2
pop_gen.set_flux_selection(flux_selector)

pop = pop_gen.draw_survey(flux_sigma=0.1)
```

```python
len(pop.distances)
```

```python
pop.n_detections
```

```python
len(pop.variability_selected[pop.variability_selected==True]) / len(pop.variability_selected)
```

```python
N_flares = [len(_) for _ in pop.flare_times]
N_flares_det = [len(_) for _ in pop.flare_times_selected]
N_assoc = len([_ for _ in pop.flare_times_selected if _ != []])
fig, ax = plt.subplots()
bins = np.linspace(0, 80)
ax.hist(N_flares, bins=bins, alpha=0.7, label="All")
ax.hist(N_flares_det, bins=bins, alpha=0.7, label="Detected")
ax.set_yscale("log")
ax.set_xlabel("Total number of flares")
ax.legend();
print("N detected flares:", sum(N_flares_det))
print("N associated sources:", N_assoc)
```

```python
d = []
for i, _ in enumerate(pop.flare_durations):
    d.extend(_)
d = np.array(d) * 52 # weeks
bins=np.linspace(min(d), max(d))
fig, ax = plt.subplots()
ax.hist(d, bins=bins, density=True)
ax.plot(bins, stats.pareto(1.5).pdf(bins), alpha=0.7, color='k', 
        label='pareto approx');
ax.set_yscale("log")
ax.set_xlabel("Flare duration (weeks)")
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

## Coincidence check with simplistic nu model

```python
from popsynth.utils.spherical_geometry import sample_theta_phi
import ligo.skymap.plot
from astropy import units as u

from cosmic_coincidence.utils.plotting import SphericalCircle
```

```python
obs_time = 7.5
```

```python
N_nu = np.random.poisson(7.1 * obs_time) 
theta, phi = sample_theta_phi(N_nu)
ra = np.rad2deg(phi)
dec = np.rad2deg(theta) - 90
nu_times = np.random.uniform(0, obs_time, N_nu)
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((7,5))
ax.scatter(pop.ra[pop.selection], pop.dec[pop.selection], 
           transform=ax.get_transform("icrs"), alpha=0.1)
match_ind = []
i = 0
for pop_r, pop_d in zip(pop.ra[pop.selection], pop.dec[pop.selection]):
    j = 0
    for r, d in zip(ra, dec):
        circle = SphericalCircle((r*u.deg, d*u.deg), 3.0*u.deg, color='r', alpha=0.5,
                             transform=ax.get_transform("icrs"))
        ax.add_patch(circle)
        if circle.contains_point((pop_r, pop_d)):
            match_ind.append((i, j))
        j+=1
    i+=1        
#ax.scatter(ra, dec, transform=ax.get_transform("icrs"))
```

```python
match_ind[0]
```

```python
fig, ax = plt.subplots()
i = 0
fts, ds = (pop.flare_times_selected[match_ind[0][0]],
           pop.flare_durations_selected[match_ind[0][0]])
if fts != []:
    for ft, d in zip(fts, ds):
        ax.plot([ft, ft+d], [i, i], color='k')
    i += 1
ax.set_xlabel("time [years]")
ax.vlines(nu_times[match_ind[0][1]], 0, 2, color='r')
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
