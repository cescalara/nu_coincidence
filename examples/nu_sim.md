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
