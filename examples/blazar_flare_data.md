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
from astropy import units as u
from scipy import stats
```

```python
hdul = fits.open('data/2fav_v10.fits')
hdul.info()
```

```python
# Get flare duration for different categories
N = len(hdul[1].data)

duration = {}
duration['bll'] = []
duration['fsrq'] = []
duration['bcu'] = []

n_flares = {}
n_flares['bll'] = []
n_flares['fsrq'] = []
n_flares['bcu'] = []

# Loop over sources
for i in range(N):
    
    c = hdul[1].data['CLASS'][i]
    
    # Get flare times corresponding to this source
    selection = np.where(hdul[2].data['FAVASRC']==i+1)[0]   
    sig_le = hdul[2].data['LESIGNIF'][selection]
    sig_he = hdul[2].data['HESIGNIF'][selection]
    condition = ((sig_le < -4) & (sig_he < -4)) | ((sig_he < -6) | (sig_le < -6))
    
    # Get non-negative flares
    non_neg_sel = selection #np.where(~condition)[0]
    tstart = hdul[2].data['TSTART'][non_neg_sel].astype(int)
    tstop = hdul[2].data['TSTOP'][non_neg_sel].astype(int)     
    
    # Merge adjacent flare periods
    eq_ind = np.where(np.equal(tstart[1:], tstop[:-1]))[0]
    a = np.delete(tstart, eq_ind+1)
    b = np.delete(tstop, eq_ind)
    d = (b-a) / 604800 # duration in weeks
    
    if c == 'bll':
        duration['bll'].extend(d)
        n_flares['bll'].append(len(d))
        
    elif c == 'fsrq':
        duration['fsrq'].extend(d)
        n_flares['fsrq'].append(len(d))
        
    elif c == 'bcu':
        duration['bcu'].extend(d)
        n_flares['bcu'].append(len(d))
```

```python
fig, ax = plt.subplots()
bins=np.linspace(1, 110)
ax.hist(duration['fsrq'], label='fsrq', alpha=0.7, bins=bins, density=True)
ax.hist(duration['bll'], label='bll', alpha=0.7, bins=bins, density=True)
ax.hist(duration['bcu'], label='bcu', alpha=0.7, bins=bins, density=True);
ax.plot(bins, stats.pareto(1.5).pdf(bins), alpha=0.7, color='k', 
        label='pareto approx');
ax.plot(bins, stats.lognorm(2.5, 0, 1e-1).pdf(bins), color='k', linestyle=':')
ax.set_yscale('log')
ax.legend()
```

```python
# Rates
time = (473615018 - 239557418) * u.s
time = time.to(u.year)

bins = 10**np.linspace(-1, 2)
fig, ax = plt.subplots()
for key, value in n_flares.items():
    print(key, min((value/time.value)))
    ax.hist((value/time.value), label=key, alpha=0.7, density=True, bins=bins)

ax.hist(stats.pareto(1.5).rvs(10000) * 0.1, alpha=0.5, color='k', bins=bins, 
        density=True)
#ax.plot(bins, stats.pareto(0.1).pdf(bins), color='k', label='pareto approx')
ax.plot(bins, stats.cauchy(0, 0.2).pdf(bins), color='r')
ax.plot(bins, stats.lognorm(2, 0, 1e-1).pdf(bins), color='k', linestyle=':')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
```

```python
max(n_flares[''])
```

```python
fig, ax = plt.subplots()
x = np.linspace(0.1, 4)
#ax.plot(x, stats.pareto(1).pdf(x))
ax.hist(stats.pareto(0.1).rvs(10000) * 0.1, bins=x)
ax.hist((np.random.pareto(0.1, 10000) + 1) * 0.1, bins=x)
ax.set_xscale('log')
ax.set_yscale('log')
```

```python
min(stats.pareto(1).rvs(10000))
```

```python
np.random.choice([True, False], p=[0.8, 0.2])
```

```python
sum(hdul[1].data['FLARES'])
```

```python
# Finding negative flares
total_neg = 0
for i in range(N):
    selection = np.where(hdul[2].data['FAVASRC'] == i+1)[0]
    N_neg = 0
    for s in selection:
        sig_le = hdul[2].data['LESIGNIF'][s]
        sig_he = hdul[2].data['HESIGNIF'][s]

        if ((sig_he <= -6) or (sig_le <= -6)) or ((sig_he <= -4) and (sig_le <= -4)):
            N_neg += 1
            
    if N_neg != hdul[1].data['NEGFLR'][i]:
        print(N_neg, hdul[1].data['NEGFLR'][i])
    total_neg += N_neg
```

```python
total_neg - sum(hdul[1].data['NEGFLR'])
```

```python

```
