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

# Blazar-neutrino connected simulation 

We demonstrate how to run a simulation for a population of blazars that produce neutrinos according to their gamma-ray fluxes in the 0.1 to 100 GeV range.

```python
from popsynth.utils.logging import quiet_mode
from icecube_tools.detector.effective_area import EffectiveArea
from icecube_tools.detector.energy_resolution import EnergyResolution
from icecube_tools.detector.angular_resolution import AngularResolution

from nu_coincidence.blazar_nu.connected import BlazarNuConnectedSim
from nu_coincidence.blazar_nu.connected import BlazarNuConnectedResults
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

The inputs to `BlazarNuConnectedSim` are specified by the `.yml` config files provided with the `nu_coincidence` package. We can check what is available using the helper function `get_available_config()`. 

```python
get_available_config()
```

The connecting flux factor, $Y_{\nu\gamma}$ can be set in the `nu_connected_...` config files or specified at run time. we may want to explore different flux factors and their implications, as we show below.

```python
flux_factors = [0.001, 0.01]
sub_file_names = ["output/test_connected_sim_%.1e.h5" % ff for ff in flux_factors]
```

For this example, we use the reference blazar model (`bllac_ref.yml` and `fsrq_ref.yml`) together with the connected IceCube HESE and EHE alerts models (`nu_connected_hese.yml` and `nu_connected_ehe.yml`). This will run for a couple of mins.

```python
for ff, sfn in zip(flux_factors, sub_file_names):
    sim = BlazarNuConnectedSim(file_name="output/test_connected_sim_%.1e.h5" % ff, 
                               N=2, # Number of sims to run
                               bllac_config="bllac_ref.yml",
                               fsrq_config="fsrq_ref.yml", 
                               nu_hese_config="nu_connected_hese.yml", 
                               nu_ehe_config="nu_connected_ehe.yml", seed=42, 
                               flux_factor=ff, 
                               flare_only=True, # only flare emission contributes
                               det_only=True) # only detected blazars contribute
    
    sim.run(parallel=False) # define number of parallel jobs to run
```

If working with different flux factors, there is a helper function to merge together the separate output files.

```python
BlazarNuConnectedResults.merge_over_flux_factor(sub_file_names, flux_factors,
                                            write_to="output/test_connected_sim.h5",
                                            delete=True)

results = BlazarNuConnectedResults(["output/test_connected_sim.h5"])
```

The results are organised to store relevant information divided into the `results.bllac` and `results.fsrq` populations. We focus on the total number of alerts, `n_alerts`, and the number of sources that produce multiple detected neutrinos, `n_multi`, from each simulated survey.

```python
results.bllac
```

```python
results.fsrq
```

```python

```
