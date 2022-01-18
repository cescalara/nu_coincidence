---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Blazar-neutrino chance coincidence simulation

We demonstrate how to run a simulation of a blazar population and diffuse neutrino flux that have no underlying connection. The simulation makes use of the configuration files provided with the package.

```python
from popsynth.utils.logging import quiet_mode
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution

from nu_coincidence.blazar_nu.coincidence import BlazarNuCoincidenceSim
from nu_coincidence.blazar_nu.coincidence import BlazarNuCoincidenceResults
from nu_coincidence.utils.package_data import get_available_config

quiet_mode() # silence popsynth logs/output
```

Firstly, we make sure that we have the necessary files for running an IceCube simulation by using the `icecube_tools` package

```python
my_aeff = EffectiveArea.from_dataset("20181018")
my_aeff = EffectiveArea.from_dataset("20131121")
my_eres = EnergyResolution.from_dataset("20150820")
my_angres = AngularResolution.from_dataset("20181018")
```

The inputs to `BlazarNuCoincidenceSim` are specified by the `.yml` config files provided with the `nu_coincidence` package. We can check what is available using the helper function `get_available_config()`. 

```python
get_available_config()
```

For this example, we can use the reference blazar model (`bllac_ref.yml` and `fsrq_ref.yml`) together with the diffuse IceCube HESE and EHE alerts models (`nu_diffuse_hese.yml` and `nu_diffuse_ehe.yml`).

```python
sim = BlazarNuCoincidenceSim(
    file_name="output/test_coincidence_sim.h5", # output file name
    N=2, # number of independent simulations to run
    bllac_config="bllac_ref.yml", 
    fsrq_config="fsrq_ref.yml",
    nu_hese_config="nu_diffuse_hese.yml",
    nu_ehe_config="nu_diffuse_ehe.yml",
    seed=42,
)

sim.run(parallel=False) # define number of parallel jobs to run
```

The results are organised to store relevant information on the number of different coincidences, divided into the `results.bllac` and `results.fsrq` populations. We study *spatial*, *variable* and *flaring* coincidences. In just 2 simulations, we expect to see spatial and variable coincidences, but flaring coincidences are unlikely. In the event that flaring coincidences occur, several other population/neutrino properties are also stored for further analysis. 

```python
results = BlazarNuCoincidenceResults.load(["output/test_coincidence_sim.h5"])
```

```python
results.bllac
```

As we ran 2 simulations, we have a coincidence count for each case.

```python
results.bllac["n_spatial"]
```

```python
results.fsrq
```
