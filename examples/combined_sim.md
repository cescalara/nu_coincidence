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

## Testing dask

```python
from dask.distributed import LocalCluster, Client
import h5py
import numpy as np
```

```python
import sys
sys.path.append("../")

from cosmic_coincidence.simulation import Simulation
```

```python
 # Initialise file
file_name = "output/test_sim.h5"
with h5py.File(file_name, "w") as f:
    f.attrs["name"] = np.string_(file_name)
```

```python
sim = Simulation(file_name=file_name, N=32)
```

```python
#cluster = LocalCluster(n_workers=6)
client = Client()
client
```

```python
sim.run(client)
```

```python
client.close()
```

```python
with h5py.File("output/test_sim.h5", "r") as f:
    for key in f:
        print(key)
```

## Old stuff

```python
import numpy as np
from matplotlib import pyplot as plt
import h5py
import ligo.skymap.plot
from astropy import units as u
```

```python
from popsynth.utils.configuration import popsynth_config
```

```python
import sys
sys.path.append("../")
from cosmic_coincidence.blazars.bllac import BLLacLDDEModel, BLLacPopWrapper
from cosmic_coincidence.blazars.fsrq import FSRQLDDEModel, FSRQPopWrapper
from cosmic_coincidence.blazars.fermi_interface import FermiPopParams
from cosmic_coincidence.utils.plotting import SphericalCircle
from cosmic_coincidence.utils.coincidence import (check_spatial_coincidence, 
                                                  check_temporal_coincidence, 
                                                  run_sim, submit_sim)
from cosmic_coincidence.populations.aux_samplers import (
    VariabilityAuxSampler,
    FlareRateAuxSampler,
    FlareTimeAuxSampler,
    FlareDurationAuxSampler,
)
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

## Blazar models


BL Lac objects

```python
bllac_ldde = BLLacLDDEModel()
bllac_ldde.A = 3.39e4
bllac_ldde.gamma1 = 0.27
bllac_ldde.Lstar = 0.28e48
bllac_ldde.gamma2 = 1.86
bllac_ldde.zcstar = 1.34
bllac_ldde.p1star = 2.24
bllac_ldde.tau = 4.92
bllac_ldde.p2 = -7.37
bllac_ldde.alpha = 4.53e-2
bllac_ldde.mustar = 2.1
bllac_ldde.beta = 6.46e-2
bllac_ldde.sigma = 0.26
bllac_ldde.Lmax = 1e50
bllac_ldde.prep_pop()
```

FSRQs

```python
fsrq_ldde = FSRQLDDEModel()
fsrq_ldde.A = 3.06e4
fsrq_ldde.gamma1 = 0.21
fsrq_ldde.Lstar = 0.84e48
fsrq_ldde.gamma2 = 1.58
fsrq_ldde.zcstar = 1.47
fsrq_ldde.p1star = 7.35
fsrq_ldde.tau = 0
fsrq_ldde.p2 = -6.51
fsrq_ldde.alpha = 0.21
fsrq_ldde.mustar = 2.44
fsrq_ldde.beta = 0
fsrq_ldde.sigma = 0.18
```

```python
# For popsynth
fsrq_ldde.Lmax = 1e50
fsrq_ldde.prep_pop()
```

## Neutrino model

```python
Emin = 1e5 # GeV
Emax = 1e8 # GeV
Emin_det = 2e5 # GeV
```

```python
# Effective area
effective_area = EffectiveArea.from_dataset("20181018")

# Energy resolution
energy_res = EnergyResolution.from_dataset("20150820")

# Angular resolution
ang_res = AngularResolution.from_dataset("20181018")

# Detector
detector = IceCube(effective_area, energy_res, ang_res)
```

```python
power_law_atmo = PowerLawFlux(2.5e-18, 1e5, 3.7, lower_energy=Emin, upper_energy=1e8)
atmospheric = DiffuseSource(flux_model=power_law_atmo)
power_law = PowerLawFlux(1.0e-18, 1e5, 2.19, lower_energy=Emin, upper_energy=1e8)
astrophysical_bg = DiffuseSource(flux_model=power_law)
sources = [atmospheric, astrophysical_bg]
```

```python
nu_simulator = Simulator(sources, detector)
nu_simulator.time = obs_time # years
nu_simulator.max_cosz = 0.1
```

## Simulate

```python
bllac_popsynth = bllac_ldde.popsynth()
variability = VariabilityAuxSampler()
variability.weight = 0.05
flare_rate = FlareRateAuxSampler()
flare_rate.xmin = 1 / 7.5
flare_rate.xmax = 15
flare_rate.index = 1.5
flare_times = FlareTimeAuxSampler()
flare_times.obs_time = 10
flare_durations = FlareDurationAuxSampler()
flare_rate.set_secondary_sampler(variability)
flare_times.set_secondary_sampler(flare_rate)
flare_durations.set_secondary_sampler(flare_times)
bllac_popsynth.add_observed_quantity(flare_durations)
```

```python
pop = bllac_popsynth.draw_survey(boundary=4e-12, hard_cut=True)
```

```python
test = pop._auxiliary_quantities["flare_durations"]["true_values"]

with h5py.File("output/test_h5.h5", "w") as f:
    f.create_dataset("test", data=test)
```

```python
with h5py.File("output/test_h5.h5", "r") as f:
    test = f["test"][()]
```

```python
pop.writeto("output/test_pop.h5")
```

## Check results

```python
with h5py.File("output/test_sim_err2_1000.h5", "r") as f:
    bllac_n_spatial = f["bllac/n_spatial"][()]
    bllac_n_variable = f["bllac/n_variable"][()
    bllac_n_flaring = f["bllac/n_flaring"][()]
    
    fsrq_n_spatial = f["fsrq/n_spatial"][()]
    fsrq_n_variable = f["fsrq/n_variable"][()]
    fsrq_n_flaring = f["fsrq/n_flaring"][()]
```

```python
fig, ax = plt.subplots()
ax.hist(bllac_n_spatial);
ax.hist(bllac_n_variable);
ax.hist(bllac_n_flaring);
```

```python
sum(bllac_n_flaring) / len(bllac_n_flaring)
```

```python
fig, ax = plt.subplots()
ax.hist(fsrq_n_spatial);
ax.hist(fsrq_n_variable);
ax.hist(fsrq_n_flaring);
```

```python
sum(fsrq_n_flaring) / len(fsrq_n_flaring)
```

```python
#fsrq_n_flaring
```

```python

```
