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
from matplotlib import pyplot as plt

import sys
sys.path.append("../")
from cosmic_coincidence.populations.sbpl_population import SBPLZPowExpCosmoPopulation
from cosmic_coincidence.distributions.sbpl_distribution import SBPLDistribution
from cosmic_coincidence.utils.interface import Ajello14PDEModel
```

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
ax.plot(z, pop.spatial_distribution.dNdV(z) * pop.spatial_distribution.differential_volume(z) / pop.spatial_distribution.time_adjustment(z))
```

```python
pop.draw_survey(boundary=1e2, no_selection=True)
```
