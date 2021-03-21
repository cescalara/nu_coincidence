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
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux, BrokenPowerLawFlux
from icecube_tools.source.source_model import DiffuseSource
from icecube_tools.simulator import Simulator
```

```python
Emin = 5e4 # GeV
```

```python
# Effective area
Aeff_filename = "input/IC86-2012-TabulatedAeff.txt"
effective_area = EffectiveArea(Aeff_filename)

# Energy resolution
eres_file = "input/effective_area.h5"
energy_res = EnergyResolution(eres_file)

# Angular resolution
Ares_file = "input/IC86-2012-AngRes.txt"
ang_res = AngularResolution(Ares_file)

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
simulator.time = 1.0 # years
simulator.max_cosz = 0.1
simulator.run(show_progress=True, seed=42)
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
ang_err = np.array(simulator.ang_err) * 3
ax.hist(ang_err);
```

```python
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((7, 5))
for ra, dec, err in zip(np.rad2deg(simulator.ra), np.rad2deg(simulator.dec), 
                        simulator.ang_err):
    circle = SphericalCircle((ra*u.deg, dec*u.deg), err*u.deg*3, 
                             transform=ax.get_transform("icrs"))
    ax.add_patch(circle)
    
    #ax.scatter(np.rad2deg(simulator.ra), np.rad2deg(simulator.dec), 
#           transform=ax.get_transform("icrs"))
```

## Scraping icecube info

Can use this to update icecube_tools.

```python
import requests
import zipfile
```

```python
# get file
url = "http://icecube.wisc.edu/data-releases/20210126_PS-IC40-IC86_VII.zip"
response = requests.get(url, stream=True)
```

```python
# save locally 
with open("data/test_dl_file.zip", 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192): 
        f.write(chunk)
```

```python
# unzip
with zipfile.ZipFile("data/test_dl_file.zip", 'r') as zip_ref:
    zip_ref.extractall("data/test_dl_file")
```