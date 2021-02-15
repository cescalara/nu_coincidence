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

```python
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import stats
```

```python
hdul = fits.open('data/2fav_v10.fits')
hdul.info()
```

```python
# Get flare duration for different categories
N = len(hdul[1].data)

duration_bll = []
duration_fsrq = []
duration_bcu = []

# Loop over sources
for i in range(N):
    
    c = hdul[1].data['CLASS'][i]
    
    # Get flare times corresponding to this source
    selection = np.where(hdul[2].data['FAVASRC']==i+1)[0]    
    tstart = hdul[2].data['TSTART'][selection].astype(int)
    tstop = hdul[2].data['TSTOP'][selection].astype(int)
    
    # Merge adjacent flare periods
    eq_ind = np.where(np.equal(tstart[1:], tstop[:-1]))[0]
    a = np.delete(tstart, eq_ind+1)
    b = np.delete(tstop, eq_ind)
    d = (b-a) / 604800 # duration in weeks
    
    if c == 'bll':
        duration_bll.extend(d)
    elif c == 'fsrq':
        duration_fsrq.extend(d)
    elif c == 'bcu':
        duration_bcu.extend(d)
```

```python
fig, ax = plt.subplots()
bins=np.linspace(1, 110)
ax.hist(duration_fsrq, label='fsrq', alpha=0.7, bins=bins, density=True)
ax.hist(duration_bll, label='bll', alpha=0.7, bins=bins, density=True)
ax.hist(duration_bcu, label='bcu', alpha=0.7, bins=bins, density=True);
ax.plot(bins, stats.pareto(1.5).pdf(bins), alpha=0.7, color='k', 
        label='pareto approx');
ax.set_yscale('log')
ax.legend()
```

```python

```
