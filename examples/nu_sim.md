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

## Scraping icecube info

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

## Running sim using icecube_tools

```python
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution
from icecube_tools.detector.detector import IceCube
from icecube_tools.source.flux_model import PowerLawFlux, BrokenPowerLawFlux
from icecube_tools.source.source_model import DiffuseSource
```

```python
# Fix loading of files
```

```python
# Effective area
Aeff_filename = FILE_STEM + "input/IC86-2012-TabulatedAeff.txt"
effective_area = EffectiveArea(Aeff_filename)

# Energy resolution
eres_file = FILE_STEM + "input/effective_area.h5"
energy_res = EnergyResolution(eres_file)

# Angular resolution
Ares_file = FILE_STEM + "input/IC86-2012-AngRes.txt"
ang_res = AngularResolution(Ares_file)

# Detector
detector = IceCube(effective_area, energy_res, ang_res)
```
