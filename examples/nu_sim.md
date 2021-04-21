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

## Running sim using icecube_tools

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
import sys
sys.path.append("../../icecube_tools/")

from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux, BrokenPowerLawFlux
from icecube_tools.source.source_model import DiffuseSource
from icecube_tools.simulator import Simulator

from icecube_tools.utils.vMF import get_kappa, get_theta_p
```

```python
kappa = get_kappa(1, 0.68)
print(kappa)
theta_p_1 = get_theta_p(kappa, 0.68)
theta_p_2 = get_theta_p(kappa, 0.95)
theta_p_3 = get_theta_p(kappa, 0.99)
print(theta_p_1, theta_p_2, theta_p_3)
```

```python
Emin = 5e4 # GeV
```

```python
# Effective area
effective_area = EffectiveArea.from_dataset("20181018")

# Energy resolution
energy_res = EnergyResolution.from_dataset("20150820")

# Angular resolution
ang_res = AngularResolution.from_dataset("20181018", ret_ang_err_p=0.9, offset=0.4)

# Detector
detector = IceCube(effective_area, energy_res, ang_res)
```

```python
power_law_atmo = PowerLawFlux(2.5e-18, 1e5, 3.7, lower_energy=Emin, upper_energy=1e8)
atmospheric = DiffuseSource(flux_model=power_law_atmo)
power_law = PowerLawFlux(1.01e-18, 1e5, 2.19, lower_energy=Emin, upper_energy=1e8)
astrophysical_bg = DiffuseSource(flux_model=power_law)
sources = [atmospheric, astrophysical_bg]
```

```python
simulator = Simulator(sources, detector)
simulator.time = 10 # years
simulator.max_cosz = 0.1
simulator.run(show_progress=True, seed=987)
```

```python
reco_energy = np.array(simulator.reco_energy)
len(reco_energy[reco_energy>4e5])
```

```python
bins = 10**np.linspace(2, 8)
fig, ax = plt.subplots()
ax.hist(simulator.true_energy, bins=bins)
ax.hist(simulator.reco_energy, bins=bins)
ax.set_xscale("log")
```

```python
fig, ax = plt.subplots()
ax.hist(np.array(simulator.true_energy)[reco_energy>4e5], bins = 10**np.linspace(4, 8))
ax.set_xscale("log")
ax.set_yscale("log")
```

```python
fig, ax = plt.subplots()
ang_err = np.array(simulator.ang_err)[reco_energy>4e5] 
ax.hist(ang_err);
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((7, 5))
for ra, dec, err in zip(np.rad2deg(simulator.ra), np.rad2deg(simulator.dec), 
                        simulator.ang_err):
    circle = SphericalCircle((ra*u.deg, dec*u.deg), err*u.deg*2, 
                             transform=ax.get_transform("icrs"))
    ax.add_patch(circle)
    
    #ax.scatter(np.rad2deg(simulator.ra), np.rad2deg(simulator.dec), 
#           transform=ax.get_transform("icrs"))
```

### Compare Aeff

```python
te_bins = effective_area.true_energy_bins
cosz_bins = effective_area.cos_zenith_bins
aeff = effective_area.values
```

```python
aeff_north = aeff
```

```python
fig, ax = plt.subplots()
red_fac = 1e-3
#ax.plot(te_bins[:-1], aeff.sum(axis=1)*red_fac)
#ax.plot(te_bins[:-1], aeff.T[cosz_bins[1:]>0].T.sum(axis=1)*red_fac)
#ax.plot(te_bins[:-1], aeff.T[cosz_bins[1:]<0].T.sum(axis=1)*red_fac)
ax.plot(te_bins[:-1], aeff.sum(axis=1)*red_fac*stats.norm(5e5, 0.3*5e5).cdf(te_bins[:-1]))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e4, 1e7)
ax.set_ylim(1e-3, 1e3)
ax.grid()
```

```python
from scipy import stats
```

```python

```

```python
np.shape(aeff.sum(axis=1))
```

```python
len(cosz_bins)
```

```python
file_stem = "/Users/fran/Downloads/20131121_Search_for_contained_neutrino_events_at_energies_above_30_TeV_in_2_years_of_data/effective_areas/"
mu_file_name = file_stem + "numu_north.txt"
e_file_name = file_stem + "nue_north.txt"
tau_file_name = file_stem + "nutau_north.txt"
```

```python
out_mu = np.loadtxt(mu_file_name, skiprows=2)
out_e = np.loadtxt(e_file_name, skiprows=2)
out_tau = np.loadtxt(tau_file_name, skiprows=2)
```

```python
fig, ax = plt.subplots()
ax.plot(out.T[0], (out_mu.T[2] + out_e.T[2] + out_tau.T[2])*0.1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e4, 1e7)
ax.set_ylim(1e-3, 1e3)
ax.grid()
```

```python
out
```

## Effective area with E selection

```python
power_law = PowerLawFlux(1.01e-18, 1e5, 1.5, lower_energy=Emin, upper_energy=1e8)
astrophysical_bg = DiffuseSource(flux_model=power_law)
sources = [astrophysical_bg]
```

```python
simulator = Simulator(sources, detector)
simulator.time = 50 # years
simulator.max_cosz = 0.1
simulator.run(show_progress=True, seed=987)
```

```python
bins = 10**np.linspace(4, 8, 100)
dN_dt_init, _ = np.histogram(simulator.true_energy, bins=bins)
dN_dt_true, _ = np.histogram(simulator.reco_energy, bins=bins)

fig, ax = plt.subplots()
ax.plot(bins[1:], dN_dt_true / dN_dt_init)
ax.set_xscale("log")
```

```python
fig, ax = plt.subplots()
ax.hist(simulator.true_energy, bins=bins);
ax.set_xscale("log")
```

## Angular resolution

```python
ang_res = AngularResolution.from_dataset("20181018", ret_ang_err_p=0.9, offset=0.0)
```

```python
fig, ax = plt.subplots()
ax.plot(ang_res.true_energy_values, ang_res.values)
ax.set_xscale("log")
ax.set_ylim(0.2, 0.5)
ax.set_xlim(1e5, 1e9)
```

```python

```

```python

```
