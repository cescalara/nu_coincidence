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

from cosmic_coincidence.coincidence.blazar_nu import BlazarNuSimulation
from cosmic_coincidence.neutrinos.icecube import IceCubeObsParams, IceCubeObsWrapper
```

```python
file_name = "output/test_sim.h5"
sim = BlazarNuSimulation(file_name=file_name, N=16)
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

```python code_folding=[]
with h5py.File("output/test_sim.h5", "r") as f:
    for key in f["survey_0/"]:
        print(key)
```

```python
# scaling 
# file size
(469808/16) * 1e5 / 1e9 
```

```python
# scaling
# time (hours)
(((120/16) * 1e5) / 100) / (60 * 60)
```

## Check coincidence stuff

```python
file_name = "output/test_sim.h5"
sim = BlazarNuSimulation(file_name=file_name, N=16)
```

```python
bllac_params = sim._bllac_param_servers[0]
fsrq_params = sim._fsrq_param_servers[0]
nu_params = sim._nu_param_servers[0]
```

```python
bllac_pop = sim._bllac_pop_wrapper(bllac_params)
fsrq_pop = sim._fsrq_pop_wrapper(fsrq_params)
nu_obs = sim._nu_obs_wrapper(nu_params)
```

```python
coincidence = sim._coincidence_check(bllac_pop, fsrq_pop, nu_obs)
```

```python
bllac_pop.write()
fsrq_pop.write()
nu_obs.write()
coincidence.write()
```

```python
with h5py.File("output/test_sim.h5", "r") as f:
    for key in f["survey_0/blazar_nu_coincidence/bllac"]:
        print(f["survey_0/blazar_nu_coincidence/bllac/n_spatial"][()])
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
