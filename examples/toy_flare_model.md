---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.0
  kernelspec:
    display_name: cosmic_coincidence
    language: python
    name: cosmic_coincidence
---

```python
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
```

```python
import sys
sys.path.append("../")
```

```python
from cosmic_coincidence.blazars.bllac import BLLacLDDEModel
from cosmic_coincidence.blazars.fsrq import FSRQLDDEModel
from cosmic_coincidence.populations.aux_samplers import (VariabilityAuxSampler, 
                                                         FlareRateAuxSampler, 
                                                         FlareTimeAuxSampler,
                                                         FlareDurationAuxSampler,
                                                         FlareAmplitudeAuxSampler,)
```

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
# For popsynth
ldde.Lmax = 1e50
ldde.prep_pop()
popsynth = ldde.popsynth()

variability = VariabilityAuxSampler()
variability.weight = 0.05

flare_rate = FlareRateAuxSampler()
flare_rate.xmin = 1/7.5
flare_rate.xmax = 15
flare_rate.index = 1.5

flare_times = FlareTimeAuxSampler()
flare_times.obs_time = 7.5 # years

flare_durations = FlareDurationAuxSampler()

flare_amplitudes = FlareAmplitudeAuxSampler()
flare_amplitudes.index = 5
flare_amplitudes.xmin=1.2

flare_rate.set_secondary_sampler(variability)
flare_times.set_secondary_sampler(flare_rate)
flare_durations.set_secondary_sampler(flare_times)
flare_amplitudes.set_secondary_sampler(flare_times)

popsynth.add_observed_quantity(flare_durations)
popsynth.add_observed_quantity(flare_amplitudes)
```

```python
pop = popsynth.draw_survey(boundary=4e-12, hard_cut=True)
```

```python
fas = []
for fa in pop.flare_amplitudes:
    if fa.size != 0:
        fas.extend(fa)
```

```python
fig, ax = plt.subplots()
ax.hist(fas)
```

```python

```
