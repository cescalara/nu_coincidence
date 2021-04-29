---
jupyter:
  jupytext:
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

## 4FGL Catalog 

```python
from matplotlib import pyplot as plt
from astropy.io import fits
import ligo.skymap.plot
```

```python
hdul = fits.open("data/gll_psc_v22.fit")
hdul.info()
```

```python
hdul[1].header
```

```python
glon = hdul[1].data["GLON"]
glat = hdul[1].data["GLAT"]
```

```python
bcu_sel = hdul[1].data["CLASS1"] == "bcu"
bll_sel = hdul[1].data["CLASS1"] == "bll"
fsrq_sel = hdul[1].data["CLASS1"] == "fsrq"
```

```python
sel = fsrq_sel
fig, ax = plt.subplots(subplot_kw={"projection": "astro degrees mollweide"})
fig.set_size_inches((12, 7))
ax.scatter(glon[sel], glat[sel], transform=ax.get_transform("galactic"))
```

```python

```
