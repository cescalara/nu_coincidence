---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: cosmic_coincidence
    language: python
    name: cosmic_coincidence
---

```python
import numpy as np
from matplotlib import pyplot as plt
import h5py
```

```python
file_name = "output/test_new_flare_model.h5"
with h5py.File(file_name, "r") as f:
    N = f.attrs["N"]
    bllac_n_spatial = np.zeros(N)
    bllac_n_variable = np.zeros(N)
    bllac_n_flaring = np.zeros(N)
    
    fsrq_n_spatial = np.zeros(N)
    fsrq_n_variable = np.zeros(N)
    fsrq_n_flaring = np.zeros(N)
    
    for i in range(N):
        
        bllac_group = f["survey_%i/blazar_nu_coincidence/bllac" % i]
        bllac_n_spatial[i] = bllac_group["n_spatial"][()]
        bllac_n_variable[i] = bllac_group["n_variable"][()]
        bllac_n_flaring[i] = bllac_group["n_flaring"][()]

        fsrq_group = f["survey_%i/blazar_nu_coincidence/fsrq" % i]
        fsrq_n_spatial[i] = fsrq_group["n_spatial"][()]
        fsrq_n_variable[i] = fsrq_group["n_variable"][()]
        fsrq_n_flaring[i] = fsrq_group["n_flaring"][()]
```

```python
total_n_spatial = bllac_n_spatial + fsrq_n_spatial
total_n_variable = bllac_n_variable + fsrq_n_variable
total_n_flaring = bllac_n_flaring + fsrq_n_flaring
```

```python
plt.style.use("minimalist")
```

```python
fig, ax = plt.subplots(2, 1)
bins = np.linspace(0, 25, 26) - 0.5
fig.set_size_inches((7, 7))
ax[0].hist(bllac_n_spatial, bins=bins, density=True, alpha=0.5, color="blue", 
        label="BL Lac");
ax[0].hist(fsrq_n_spatial, bins=bins, density=True, alpha=0.5, color="green", 
        label="FSRQ");
ax[0].hist(total_n_spatial, bins=bins, density=True, histtype="step", color="k", lw=3,
           label="Total")
ax[0].set_xticks(bins[::2] + 0.5);
ax[0].legend()
ax[0].set_xlabel("Number of coincidences")
ax[0].set_title("Spatial")
bins = np.linspace(0, 10, 11) - 0.5
ax[1].hist(bllac_n_variable, bins=bins, density=True, alpha=0.5, color="blue", 
        label="BL Lac");
ax[1].hist(fsrq_n_variable, bins=bins, density=True, alpha=0.5, color="green", 
        label="FSRQ");
ax[1].hist(total_n_variable, bins=bins, density=True, histtype="step", color="k", lw=3,
           label="Total")
ax[1].set_xticks(bins + 0.5);
ax[1].set_title("Variable")
ax[1].set_xlabel("Number of coincidences")
fig.tight_layout()
fig.savefig("figures/coincidence_dist.pdf", bbox_inches="tight", dpi=200)
```

```python

```

```python

```

```python
print("BL Lac flare rate: %.2f" % ((len(bllac_n_flaring[bllac_n_flaring>=1])/N)*100))
print("FSRQ flare rate: %.2f" % ((len(fsrq_n_flaring[fsrq_n_flaring>=1])/N)*100))
print("Total flare rate: %.2f" % ((len(total_n_flaring[total_n_flaring >= 1])/N)*100))
```

```python
fig, ax = plt.subplots()
labels = ["spatial", "variable", "flaring"]
bllac_data = [bllac_n_spatial/N, bllac_n_variable/N, bllac_n_flaring/N]
ax.bar([0, 1, 2], [1, 2, 3], tick_label=labels)
```

```python

```
