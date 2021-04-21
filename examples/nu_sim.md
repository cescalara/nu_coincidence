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
```

```python
ehe_aeff = EffectiveArea.from_dataset("20181018")
hese_aeff = EffectiveArea.from_dataset("20131121", scale_factor=0.2)

energy_res = EnergyResolution.from_dataset("20150820")
ang_res = AngularResolution.from_dataset("20181018", ret_ang_err_p=0.9, offset=0.9)

ehe_detector = IceCube(ehe_aeff, energy_res, ang_res)
hese_detector = IceCube(hese_aeff, energy_res, ang_res)
```

```python
Emin = 1e4 # GeV
Emax = 1e8 # GeV

power_law_atmo = PowerLawFlux(2.5e-18, 1e5, 3.7, lower_energy=Emin, 
                              upper_energy=Emax)
atmospheric = DiffuseSource(flux_model=power_law_atmo)
power_law = PowerLawFlux(1.0e-18, 1e5, 2.5, lower_energy=Emin, upper_energy=Emax)
astrophysical_bg = DiffuseSource(flux_model=power_law)
sources = [atmospheric, astrophysical_bg]
```

```python
simulator = Simulator(sources, hese_detector)
simulator.time = 7.5 # years
simulator.max_cosz = 1
simulator._get_expected_number()
simulator._Nex
simulator.run(show_progress=True, seed=42)
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
ang_err = np.array(simulator.ang_err)
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
```

```python
Emin = 5e4 # GeV
Emax = 1e8 # GeV

power_law_atmo = PowerLawFlux(2.5e-18, 1e5, 3.7, lower_energy=Emin, 
                              upper_energy=Emax)
atmospheric = DiffuseSource(flux_model=power_law_atmo)
power_law = PowerLawFlux(0.3e-18, 1e5, 2.5, lower_energy=Emin, upper_energy=Emax)
astrophysical_bg = DiffuseSource(flux_model=power_law)
sources = [atmospheric, astrophysical_bg]
```

```python
simulator = Simulator(sources, ehe_detector)
simulator.time = 7.5 # years
simulator.max_cosz = 1
simulator._get_expected_number()
simulator._Nex
simulator.run(show_progress=True, seed=42)
```

```python
reco_energy = np.array(simulator.reco_energy)
len(reco_energy[reco_energy>3e5])
```

```python
np.array(simulator.source_label)[reco_energy>2e5]
```

## Aeff for HESE/EHE alerts

Use public info to recreate the published Aeff in Collaboration, I. et al. The IceCube realtime alert system. Astropart Phys 92, 30–41 (2017).

* HESE: Use the public Aeff from https://icecube.wisc.edu/data-releases/2013/11/search-for-contained-neutrino-events-at-energies-above-30-tev-in-2-years-of-data/. Sum over all flavours, and reduce to account for only tracks, not showers.

* EHE: Use the muon neutrino track Aeff from https://icecube.wisc.edu/data-releases/2018/10/all-sky-point-source-icecube-data-years-2010-2012/. Add a reco energy cut to reflect the tighter constraints.

```python
import h5py
```

### HESE

```python
# HESE Aeffs
file_stem = "/Users/fran/Downloads/20131121_Search_for_contained_neutrino_events_at_energies_above_30_TeV_in_2_years_of_data/effective_areas/"
mu_north_file_name = file_stem + "numu_north.txt"
e_north_file_name = file_stem + "nue_north.txt"
tau_north_file_name = file_stem + "nutau_north.txt"
mu_south_file_name = file_stem + "numu_south.txt"
e_south_file_name = file_stem + "nue_south.txt"
tau_south_file_name = file_stem + "nutau_south.txt"
mu_file_name = file_stem + "numu_4pi.txt"
e_file_name = file_stem + "nue_4pi.txt"
tau_file_name = file_stem + "nutau_4pi.txt"
```

```python
mu_north = np.loadtxt(mu_north_file_name, skiprows=2)
e_north = np.loadtxt(e_north_file_name, skiprows=2)
tau_north = np.loadtxt(tau_north_file_name, skiprows=2)
mu_south = np.loadtxt(mu_south_file_name, skiprows=2)
e_south = np.loadtxt(e_south_file_name, skiprows=2)
tau_south = np.loadtxt(tau_south_file_name, skiprows=2)
mu = np.loadtxt(mu_file_name, skiprows=2)
e = np.loadtxt(e_file_name, skiprows=2)
tau = np.loadtxt(tau_file_name, skiprows=2)
```

### EHE

```python
effective_area = EffectiveArea.from_dataset("20181018")
energy_res = EnergyResolution.from_dataset("20150820")
ang_res = AngularResolution.from_dataset("20181018", ret_ang_err_p=0.9, offset=0.4)
detector = IceCube(effective_area, energy_res, ang_res)

power_law = PowerLawFlux(1.01e-18, 1e5, 1.1, lower_energy=1e4, upper_energy=1e7)
astrophysical_bg = DiffuseSource(flux_model=power_law)
sources = [astrophysical_bg]
```

```python
#simulator = Simulator(sources, detector)
#simulator.time = 100 # years
#simulator.max_cosz = 1
#simulator.run(show_progress=True, seed=987, N=int(1e5))
#simulator.save("output/sim_for_ehe_aeff.h5")
```

```python
with h5py.File("output/sim_for_ehe_aeff.h5") as f:
    reco_samples = f["reco_energy"][()]
    true_samples = f["true_energy"][()]
    dec_samples = f["dec"][()]
    
pl_samples = astrophysical_bg.flux_model.sample(1000000)
Ereco_min = 2e5
selection = reco_samples > Ereco_min
north_selection = (reco_samples > Ereco_min) & (dec_samples > 0)
south_selection = (reco_samples > Ereco_min) & (dec_samples < 0)
```

```python
bins = 10**np.linspace(4, 7, 50)
Einit, _ = np.histogram(pl_samples, bins=bins, density=True)
Etrue, _ = np.histogram(true_samples, bins=bins, density=True)
Etrue_sel, _ = np.histogram(true_samples[selection], bins=bins, density=True)
Etrue_sel_north, _ = np.histogram(true_samples[north_selection], bins=bins, 
                                  density=True)
Etrue_sel_south, _ = np.histogram(true_samples[south_selection], bins=bins, 
                                  density=True)
```

### Figure 7

```python
hese_aeff_fac = 0.2
ehe_aeff_fac = 20
```

```python
fig, ax = plt.subplots(3, 1)
fig.set_size_inches((7, 15))

ax[0].step(mu.T[0] * 1e-3,  (mu.T[2] + e.T[2] + tau.T[2]) * hese_aeff_fac, 
          label="HESE total", color="green")
ax[0].step(bins[1:] * 1e-3, (Etrue_sel / Einit) * ehe_aeff_fac, 
           label="EHE total", color="grey")

ax[1].step(mu_south.T[0] * 1e-3, (mu_south.T[2] + e_south.T[2] + tau_south.T[2]) 
           * hese_aeff_fac, label="HESE south", color="green")
ax[1].step(bins[1:] * 1e-3, (Etrue_sel_south / Einit) * ehe_aeff_fac, 
           label="EHE south", color="grey")

ax[2].step(mu_north.T[0] * 1e-3, (mu_north.T[2] + e_north.T[2] + tau_north.T[2]) 
           * hese_aeff_fac, label="HESE north", color="green")
ax[2].step(bins[1:] * 1e-3, (Etrue_sel_north / Einit) * ehe_aeff_fac, 
           label="EHE north", color="grey")

for axis in ax:
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Neutrino Energy [TeV]")
    axis.set_ylabel("Neutrino Effective Area [m^2]")
    axis.set_xlim(1e1, 1e4)
    axis.set_ylim(1e-3, 1e3)
    axis.grid()
    axis.legend()
fig.savefig("figures/realtime_aeff_fig7.pdf", bbox_inches="tight", dpi=100)
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

## Reading HESE Aeffs

```python
import numpy as np
import os
from matplotlib import pyplot as plt
import sys
sys.path.append("../../icecube_tools/")

from icecube_tools.detector.effective_area import R2013AeffReader, EffectiveArea
from icecube_tools.utils.data import IceCubeData, find_folders
```

```python
aeff = EffectiveArea.from_dataset("20131121")
```

```python
fig, ax = plt.subplots()
ax.pcolormesh(aeff.true_energy_bins, aeff.cos_zenith_bins, 
             np.log10(aeff.values.T))
ax.set_xscale("log")
```

```python
my_data = IceCubeData()
```

```python
my_data.ls()
```

```python

```
